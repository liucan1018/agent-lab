import unittest

from pydantic import ValidationError
from llm.models import Message, Response, ToolCall, TurnOutcome, normalize_finish_reason


def wire(arguments, **overrides):
    result = {"id": "call-1", "type": "function", "function": {"name": "lookup", "arguments": arguments}}
    result.update(overrides)
    return result


class ModelTests(unittest.TestCase):
    def test_valid_tool_call_round_trip(self):
        call = ToolCall.model_validate(wire('{"city":"北京","x":[1,true,null]}'))
        self.assertIsNone(call.error)
        self.assertEqual(ToolCall.model_validate(call.to_wire_dict()).args, call.args)

    def test_invalid_arguments_preserved_and_cannot_be_serialized(self):
        for arguments in ['{bad', '', '[]', '{"x": NaN}', '{"x": 1e999}', None, {"x": object()}]:
            with self.subTest(arguments=arguments):
                call = ToolCall.model_validate(wire(arguments))
                self.assertTrue(call.error)
                self.assertEqual(call.raw_arguments, arguments)
                with self.assertRaisesRegex(ValueError, 'Cannot serialize'):
                    call.to_wire_dict()

    def test_missing_identifiers_rejected(self):
        for data in [wire('{}', id=''), {'function': {'name': 'f', 'arguments': '{}'}}, wire('{}', function={'name': ' ', 'arguments': '{}'})]:
            with self.subTest(data=data), self.assertRaises(ValidationError):
                ToolCall.model_validate(data)

    def test_non_json_internal_values_and_mutation_are_rejected(self):
        with self.assertRaises(ValidationError):
            ToolCall(id='1', name='f', args={'x': float('nan')})
        call = ToolCall(id='1', name='f')
        call.args['x'] = object()
        with self.assertRaises(ValueError):
            call.to_wire_dict()

    def test_cyclic_and_non_string_key_rejected(self):
        value = []
        value.append(value)
        for args in ({'x': value}, {'x': {1: 'v'}}):
            with self.assertRaises(ValidationError):
                ToolCall(id='1', name='f', args=args)

    def test_outcome(self):
        cases = [(None, False, TurnOutcome.UNKNOWN), (None, True, TurnOutcome.UNKNOWN),
                 ('new_reason', True, TurnOutcome.UNKNOWN), ('stop', True, TurnOutcome.TOOL_CALLS),
                 ('tool_calls', False, TurnOutcome.TOOL_CALLS), ('length', True, TurnOutcome.LENGTH),
                 ('PROVIDER_ERROR', False, TurnOutcome.PROVIDER_ERROR)]
        for reason, has_calls, expected in cases:
            with self.subTest(reason=reason, has_calls=has_calls):
                self.assertEqual(normalize_finish_reason(reason, has_calls), expected)

    def test_response_preserves_provenance_and_null_content(self):
        msg = Response(content=None, model='example', finish_reason='new_reason').to_message()
        self.assertEqual(msg.content, '')
        self.assertEqual(msg.metadata, {'model': 'example', 'finish_reason': 'new_reason', 'stop_reason': 'unknown'})

    def test_image_and_text_content(self):
        message = Message(role='user', content=[{'type': 'image', 'data': ' x '}, {'type': 'text', 'text': 'hi'}])
        self.assertEqual(message.content[0]['mime_type'], 'image/png')
        self.assertEqual(message.text_content, 'hi')
        with self.assertRaises(ValidationError):
            Message(role='user', content=[{'type': 'image', 'data': ' '}])


if __name__ == '__main__':
    unittest.main()
