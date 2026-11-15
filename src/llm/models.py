from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Any, Literal

from typing_extensions import NotRequired, TypedDict
from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from enum import Enum

import json
import math


def _validate_json_value(value: Any, active: set[int] | None = None) -> None:
    """Reject values that cannot be represented faithfully as JSON."""
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("JSON numbers must be finite")
        return
    if type(value) not in (dict, list):
        raise ValueError(f"Unsupported JSON value: {type(value).__name__}")
    if active is None:
        active = set()
    if id(value) in active:
        raise ValueError("JSON values cannot contain cycles")
    active.add(id(value))
    try:
        if isinstance(value, dict):
            for key, item in value.items():
                if not isinstance(key, str):
                    raise ValueError("JSON object keys must be strings")
                _validate_json_value(item, active)
        else:
            for item in value:
                _validate_json_value(item, active)
    finally:
        active.remove(id(value))


class TextContent(TypedDict):
    type: Literal["text"]
    text: str


class ImageContent(TypedDict):
    type: Literal["image"]
    mime_type: Literal["image/png", "image/jpeg", "image/gif", "image/webp"]
    data: NotRequired[str]
    asset_path: NotRequired[str]
    sha256: NotRequired[str]
    width: NotRequired[int]
    height: NotRequired[int]
    size_bytes: NotRequired[int]


def normalize_image(value: Any) -> Any:
    """在 Pydantic 校验 ImageContent 前处理原始输入。"""
    if not isinstance(value, Mapping):
        return value

    image = dict(value)  # 不直接修改调用方传入的字典

    mime = image.get("mime_type", "image/png")
    if isinstance(mime, str):
        mime = mime.strip().lower()
        image["mime_type"] = "image/jpeg" if mime == "image/jpg" else mime

    for key in ("data", "asset_path"):
        raw = image.get(key)
        if isinstance(raw, str):
            raw = raw.strip()
            if raw:
                image[key] = raw
            else:
                image.pop(key)

    return image


def validate_image(image: ImageContent) -> ImageContent:
    """在字段类型校验完成后检查图片来源。"""
    if not image.get("data") and not image.get("asset_path"):
        raise ValueError("Image block must contain non-empty 'data' or 'asset_path'")
    return image


ImageBlock = Annotated[
    ImageContent,
    BeforeValidator(normalize_image),
    AfterValidator(validate_image),
]

# 根据 type 直接选择要校验的分支，错误信息也更聚焦。
ContentBlock = Annotated[
    TextContent | ImageBlock,
    Field(discriminator="type"),
]

MessageContent = str | list[ContentBlock]


class Message(BaseModel):
    role: Literal["system", "developer", "user", "assistant", "tool"]
    content: MessageContent
    metadata: dict[str, Any] | None = None

    @property
    def text_content(self) -> str:
        """纯文字视图。"""
        if isinstance(self.content, str):
            return self.content

        return "".join(part["text"] for part in self.content if part["type"] == "text")


class ToolCall(BaseModel):
    """
    统一的工具调用对象。

    内部格式：
        {
            "id": "call_001",
            "name": "get_weather",
            "args": {"city": "北京"},
            "error": None,
        }

    外部格式：
        {
            "id": "call_001",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": '{"city": "北京"}',
            },
        }

    外部格式中的 arguments 是 JSON 字符串；
    内部格式中的 args 是 Python 字典。
    """

    id: str
    name: str
    args: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    raw_arguments: Any = None

    @field_validator("id", "name")
    @classmethod
    def _nonempty_identifier(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be empty")
        return value

    @field_validator("args")
    @classmethod
    def _valid_json_args(cls, value: dict[str, Any]) -> dict[str, Any]:
        _validate_json_value(value)
        return value

    @staticmethod
    def _parse_arguments(raw: Any) -> tuple[dict[str, Any], str | None]:
        """将工具参数转换为字典，并返回解析错误。"""

        # 参数已经是字典，不需要再解析
        if isinstance(raw, dict):
            try:
                _validate_json_value(raw)
            except ValueError as exc:
                return {}, f"Invalid JSON arguments: {exc}"
            return raw, None

        # 不支持其他非字符串类型
        if not isinstance(raw, str):
            return {}, (
                "Tool arguments must be a JSON string or dict, "
                f"got {type(raw).__name__}"
            )

        # 空字符串不是合法的 JSON 参数
        if not raw.strip():
            return {}, "Malformed JSON arguments: empty string"

        try:
            parsed = json.loads(raw)
        except ValueError as exc:
            return {}, f"Malformed JSON arguments: {exc}"

        # 工具参数必须是 JSON 对象，不能是数组、数字等
        if not isinstance(parsed, dict):
            return {}, (
                "Tool arguments must be a dict, " f"got {type(parsed).__name__}"
            )

        try:
            _validate_json_value(parsed)
        except ValueError as exc:
            return {}, f"Invalid JSON arguments: {exc}"
        return parsed, None

    @model_validator(mode="before")
    @classmethod
    def _normalize_wire_dict(cls, data: Any) -> Any:
        """
        将外部工具调用格式转换成内部格式。

        如果输入已经是内部格式，则直接交给 Pydantic 校验。
        """
        if not (isinstance(data, dict) and "function" in data and "name" not in data):
            return data

        fn = data["function"]

        # function 可以是字典，也可以是具有相应属性的对象
        if isinstance(fn, dict):
            name = fn.get("name", "")
            raw_args = fn.get("arguments")
        else:
            name = getattr(fn, "name", "")
            raw_args = getattr(fn, "arguments", None)

        args, error = cls._parse_arguments(raw_args)

        return {
            "id": data.get("id", ""),
            "name": name,
            "args": args,
            "error": error,
            "raw_arguments": raw_args if error else None,
        }

    def to_wire_dict(self) -> dict[str, Any]:
        """转换为带有 JSON 字符串参数的外部工具调用格式。"""
        if self.error is not None:
            raise ValueError(f"Cannot serialize invalid tool call: {self.error}")
        _validate_json_value(self.args)  # args 是可变对象，序列化前重新检查。
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": json.dumps(
                    self.args,
                    ensure_ascii=False,
                    allow_nan=False,
                ),
            },
        }


class TurnOutcome(str, Enum):
    """Provider 中立的单轮终止原因（对标 pig-llm）。"""

    COMPLETED = "completed"
    TOOL_CALLS = "tool_calls"
    LENGTH = "length"
    CONTENT_FILTER = "content_filter"
    ABORTED = "aborted"
    PROVIDER_ERROR = "provider_error"
    UNKNOWN = "unknown"


_COMPLETED_REASONS = {
    "stop",
    "end_turn",
    "complete",
    "completed",
    "stop_sequence",
    "natural",
    "success",
    "done",
}
_TOOL_REASONS = {"tool_calls", "tool_use", "function_call", "function_calls"}
_LENGTH_REASONS = {
    "length",
    "max_tokens",
    "max_output_tokens",
    "model_length",
    "token_limit",
}
_FILTER_REASONS = {"content_filter", "safety", "blocked", "recitation", "prohibited"}
_ABORT_REASONS = {"aborted", "cancelled", "canceled", "interrupt", "interrupted"}
_ERROR_REASONS = {"error", "failed", "failure", "provider_error"}


def normalize_finish_reason(
    reason: str | None, has_tool_calls: bool = False
) -> TurnOutcome:
    """归一化各厂商私有 finish_reason 为中立枚举。"""
    if reason is None:
        return TurnOutcome.UNKNOWN
    norm = reason.strip().lower().rsplit(".", maxsplit=1)[-1]
    if norm in _COMPLETED_REASONS:
        return TurnOutcome.TOOL_CALLS if has_tool_calls else TurnOutcome.COMPLETED
    if norm in _TOOL_REASONS:
        return TurnOutcome.TOOL_CALLS
    if norm in _LENGTH_REASONS:
        return TurnOutcome.LENGTH
    if norm in _FILTER_REASONS:
        return TurnOutcome.CONTENT_FILTER
    if norm in _ABORT_REASONS:
        return TurnOutcome.ABORTED
    if norm in _ERROR_REASONS:
        return TurnOutcome.PROVIDER_ERROR
    return TurnOutcome.UNKNOWN


class Response(BaseModel):
    """统一响应：文本 + 工具调用 + usage + reasoning。"""

    model_config = ConfigDict(extra="forbid")

    content: str
    model: str
    tool_calls: list[ToolCall] | None = None
    reasoning_content: str | None = None
    usage: dict[str, int] | None = None
    finish_reason: str | None = None

    @field_validator("content", mode="before")
    @classmethod
    def _normalize_null_content(cls, value: Any) -> Any:
        return "" if value is None else value

    @property
    def outcome(self) -> TurnOutcome:
        """中立化的终止状态。"""
        return normalize_finish_reason(self.finish_reason, bool(self.tool_calls))

    def to_message(
        self,
        role: Literal["system", "developer", "user", "assistant", "tool"] = "assistant",
        stop_reason: str | None = None,
    ) -> Message:
        """将模型层完整响应直接转换为标准 Message 实体，彻底消除调度层手动累加拼装。"""
        meta: dict[str, Any] = {}
        meta["model"] = self.model
        if self.finish_reason is not None:
            meta["finish_reason"] = self.finish_reason
        if self.tool_calls:
            meta["tool_calls"] = [
                tc.model_dump() if hasattr(tc, "model_dump") else tc
                for tc in self.tool_calls
            ]
        if self.usage:
            meta["usage"] = self.usage
        if self.reasoning_content:
            meta["reasoning_content"] = self.reasoning_content
        effective_stop = stop_reason or self.outcome.value
        if effective_stop:
            meta["stop_reason"] = effective_stop
        return Message(role=role, content=self.content, metadata=meta if meta else None)


class StreamChunk(BaseModel):
    """流式增量块：文本增量 + 末块携带完整 tool_calls 与终态已拼装好的 Response。"""

    content: str
    finish_reason: str | None = None
    tool_calls: list[ToolCall] | None = None
    usage: dict[str, int] | None = None
    metadata: dict[str, Any] | None = None
    response: Response | None = None

    @property
    def outcome(self) -> TurnOutcome | None:
        if self.finish_reason is None and not self.tool_calls:
            return None
        return normalize_finish_reason(self.finish_reason, bool(self.tool_calls))
