# LLM 数据模型维护说明

本文记录 `src/llm/models.py` 的内部约定。供应商适配器应在创建这些对象前完成供应商专有格式转换；修改约定时请同步更新本文和 `tests/test_llm_models.py`。

## 类与接口速查

### 内容类型

| 类型 | 字段或定义 | 用途 |
| --- | --- | --- |
| `TextContent` | `type: Literal["text"]`、`text: str` | 文本内容块。 |
| `ImageContent` | `type: Literal["image"]`、`mime_type`；可选 `data`、`asset_path`、`sha256`、`width`、`height`、`size_bytes` | 图片内容块；支持 PNG、JPEG、GIF 和 WebP。 |
| `ImageBlock` | 带前置和后置校验的 `ImageContent` | 规范化图片输入并检查图片来源。 |
| `ContentBlock` | `TextContent \| ImageBlock` | 通过 `type` 区分文本和图片块。 |
| `MessageContent` | `str \| list[ContentBlock]` | 纯文本或多模态内容块列表。 |

`normalize_image(value)` 复制映射输入并规范化 MIME 类型和图片来源字符串；`validate_image(image)` 检查 `data` 或 `asset_path` 至少有一个非空值，否则抛出 `ValueError`。这两个函数由 `ImageBlock` 的 Pydantic 校验流程调用。

### `Message`

Pydantic 消息模型：

| 字段或接口 | 类型 | 说明 |
| --- | --- | --- |
| `role` | `Literal["system", "developer", "user", "assistant", "tool"]` | 消息角色。 |
| `content` | `MessageContent` | 消息内容。 |
| `metadata` | `dict[str, Any] \| None` | 可选元数据，默认为 `None`。 |
| `text_content` | 属性，返回 `str` | 纯文字视图；按顺序拼接文本块，忽略图片。 |

### `ToolCall`

Pydantic 工具调用模型：

| 字段或接口 | 类型 | 说明 |
| --- | --- | --- |
| `id`、`name` | `str` | 调用 ID 和工具名，不能全为空白。 |
| `args` | `dict[str, Any]` | 工具参数，默认为空字典，必须是合法 JSON 对象。 |
| `error` | `str \| None` | 外部参数解析失败时的错误信息。 |
| `raw_arguments` | `Any` | 外部参数解析失败时保留的原始参数。 |
| `to_wire_dict()` | 返回 `dict[str, Any]` | 输出外部工具调用格式，将参数序列化为 JSON 字符串；有 `error` 时抛出 `ValueError`。 |

创建时既可传入内部格式，也可传入外部格式：

```python
ToolCall(id="call_001", name="get_weather", args={"city": "北京"})

ToolCall.model_validate({
    "id": "call_001",
    "type": "function",
    "function": {"name": "get_weather", "arguments": '{"city": "北京"}'},
})
```

### `TurnOutcome` 与 `normalize_finish_reason`

`TurnOutcome` 是单轮终止原因的字符串枚举：

| 成员 | 值 | 含义 |
| --- | --- | --- |
| `COMPLETED` | `completed` | 正常完成。 |
| `TOOL_CALLS` | `tool_calls` | 产生工具调用。 |
| `LENGTH` | `length` | 达到长度或 token 限制。 |
| `CONTENT_FILTER` | `content_filter` | 内容过滤。 |
| `ABORTED` | `aborted` | 中止或取消。 |
| `PROVIDER_ERROR` | `provider_error` | 供应商错误。 |
| `UNKNOWN` | `unknown` | 未知原因。 |

`normalize_finish_reason(reason: str | None, has_tool_calls: bool = False) -> TurnOutcome` 将供应商的终止原因映射为上述枚举。

### `Response`

Pydantic 统一响应模型，禁止未声明的额外字段：

| 字段或接口 | 类型 | 说明 |
| --- | --- | --- |
| `content` | `str` | 响应文本；输入为 `None` 时转为空字符串。 |
| `model` | `str` | 模型名称。 |
| `tool_calls` | `list[ToolCall] \| None` | 可选工具调用列表。 |
| `reasoning_content` | `str \| None` | 可选推理内容。 |
| `usage` | `dict[str, int] \| None` | 可选用量信息。 |
| `finish_reason` | `str \| None` | 供应商原始终止原因。 |
| `outcome` | 属性，返回 `TurnOutcome` | 根据终止原因和是否有工具调用得到统一状态。 |
| `to_message(role="assistant", stop_reason=None)` | 返回 `Message` | 转换为消息，并将响应信息写入 `metadata`。 |

## 消息和图片

`Message.content` 可以是字符串或带 `type` 的文本、图片块列表。`text_content` 按顺序直接拼接文本块，不插入分隔符，也不包含图片。图片的 MIME 类型默认 `image/png`，`image/jpg` 归一化为 `image/jpeg`；`data` 和 `asset_path` 至少有一个非空。两者同时出现时模型不决定优先级，适配器必须根据目标协议明确处理。模型不验证图片数据、文件存在性、摘要或尺寸。

## 工具调用

`ToolCall` 支持内部的 `id`、`name`、`args` 和外部的 `function.name`、`function.arguments` 格式。ID 和名称必须非空；外部参数必须存在，且是 JSON 对象字符串或字典。参数仅接受 JSON 值、字符串对象键和有限数字；内部参数在创建和序列化时校验。可变的 `args` 在模型创建后仍能被修改，因此执行前也应检查其有效性。

解析失败的外部参数不会抛弃整个响应：模型保留 `error` 和 `raw_arguments`，`args` 为占位空字典。**调度器必须在执行前检查 `error is None`；有错误时绝不能执行占位参数。** 模型本身不执行工具，不能替代调度层检查；`to_wire_dict()` 会拒绝序列化错误调用。`raw_arguments` 可能包含敏感数据，记录日志时应按项目的数据处理规则处理。

## 结束状态和响应

`normalize_finish_reason` 将已知原因映射为 `TurnOutcome`。缺失或未知原因映射为 `UNKNOWN`，即使响应包含工具调用也不推断其成功结束；已知 `stop` 等正常结束原因伴随工具调用时映射为 `TOOL_CALLS`。长度限制、过滤、中止和供应商错误保留各自状态。供应商增加原因时，应检查其语义后更新映射和测试，不能把未知状态当成成功。需要判断是否存在调用时可单独检查 `tool_calls`，但仍需检查终止状态和每个调用的错误。

`Response.content=None` 归一化为空字符串，便于接收纯工具调用响应。`to_message()` 的 `metadata` 包含 `model`、原始 `finish_reason`（存在时）、归一化的 `stop_reason`，以及存在时的工具调用、usage 和 reasoning。传入 `stop_reason` 可以显式覆盖归一化值；应保留原始原因用于排查。

## 兼容性与验证

这些约定会改变旧行为：缺失终止原因现在是 `UNKNOWN`；空 ID 或名称会被拒绝；有解析错误的工具调用不能转换为外部格式；非标准 JSON 参数会被拒绝。接入新供应商或升级协议时，应验证完整响应、工具调用和失败路径，并在项目根目录运行：

```powershell
$env:PYTHONPATH = "src"
.venv\Scripts\python -m pytest tests/test_llm_models.py
```
