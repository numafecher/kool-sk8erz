#!/usr/bin/env python
# -*- coding: utf-8 -*-
import struct
import numpy as np

# Prioritize official TensorFlow Lite / LiteRT runtimes if installed
_TFLiteInterpreter = None
try:
    from tflite_runtime.interpreter import Interpreter as _TFLiteInterpreter
except ImportError:
    try:
        from ai_edge_litert.interpreter import Interpreter as _TFLiteInterpreter
    except ImportError:
        try:
            from tensorflow.lite import Interpreter as _TFLiteInterpreter
        except ImportError:
            _TFLiteInterpreter = None


class NumPyTFLiteInterpreter(object):
    """
    Pure-Python + NumPy fallback interpreter for TensorFlow Lite (.tflite) models.
    Parses TFLite FlatBuffers and executes inference without requiring
    TensorFlow or tflite_runtime packages.
    """
    def __init__(self, model_path=None, model_content=None, num_threads=1):
        self.model_path = model_path
        self.num_threads = num_threads
        if model_content is not None:
            self._buf = model_content
        elif model_path is not None:
            with open(model_path, 'rb') as f:
                self._buf = f.read()
        else:
            raise ValueError("Either model_path or model_content must be provided.")

        self._tensors = {}
        self._tensor_meta = []
        self._input_indices = []
        self._output_indices = []
        self._ops = []
        self._parse()

    def _get_u32(self, offset):
        return struct.unpack_from('<I', self._buf, offset)[0]

    def _get_i32(self, offset):
        return struct.unpack_from('<i', self._buf, offset)[0]

    def _get_u16(self, offset):
        return struct.unpack_from('<H', self._buf, offset)[0]

    def _get_field_offset(self, table_offset, field_index):
        vtable_offset = table_offset - self._get_i32(table_offset)
        vtable_size = self._get_u16(vtable_offset)
        entry_offset = 4 + field_index * 2
        if entry_offset < vtable_size:
            field_offset = self._get_u16(vtable_offset + entry_offset)
            if field_offset != 0:
                return table_offset + field_offset
        return None

    def _get_string(self, offset):
        str_vec = offset + self._get_u32(offset)
        str_len = self._get_u32(str_vec)
        return self._buf[str_vec + 4 : str_vec + 4 + str_len].decode('utf-8', errors='ignore')

    def _parse(self):
        root_offset = self._get_u32(0)

        # 1. Parse buffers (Model field 4)
        buffers_field = self._get_field_offset(root_offset, 4)
        buffers = []
        if buffers_field:
            buffers_vec = buffers_field + self._get_u32(buffers_field)
            num_buffers = self._get_u32(buffers_vec)
            for i in range(num_buffers):
                buf_elem = buffers_vec + 4 + i * 4
                buf_table = buf_elem + self._get_u32(buf_elem)
                data_field = self._get_field_offset(buf_table, 0)
                if data_field:
                    data_vec = data_field + self._get_u32(data_field)
                    data_len = self._get_u32(data_vec)
                    buffers.append(self._buf[data_vec + 4 : data_vec + 4 + data_len])
                else:
                    buffers.append(b'')

        # 2. Parse operator codes (Model field 1)
        opcodes_field = self._get_field_offset(root_offset, 1)
        opcodes = []
        if opcodes_field:
            opcodes_vec = opcodes_field + self._get_u32(opcodes_field)
            num_opcodes = self._get_u32(opcodes_vec)
            for i in range(num_opcodes):
                op_elem = opcodes_vec + 4 + i * 4
                op_table = op_elem + self._get_u32(op_elem)
                bc_field = self._get_field_offset(op_table, 3) or self._get_field_offset(op_table, 0)
                code = struct.unpack_from('<b', self._buf, bc_field)[0] if bc_field else 0
                opcodes.append(code)

        # 3. Parse first subgraph (Model field 2)
        subgraphs_field = self._get_field_offset(root_offset, 2)
        if not subgraphs_field:
            raise ValueError("No subgraphs found in TFLite model.")
        subgraphs_vec = subgraphs_field + self._get_u32(subgraphs_field)
        sg_table = subgraphs_vec + 4 + self._get_u32(subgraphs_vec + 4)

        # SubGraph tensors (field 0)
        tensors_field = self._get_field_offset(sg_table, 0)
        tensors_vec = tensors_field + self._get_u32(tensors_field)
        num_tensors = self._get_u32(tensors_vec)

        self._tensor_meta = []
        for i in range(num_tensors):
            t_elem = tensors_vec + 4 + i * 4
            t_table = t_elem + self._get_u32(t_elem)
            shape_field = self._get_field_offset(t_table, 0)
            shape = []
            if shape_field:
                shape_vec = shape_field + self._get_u32(shape_field)
                for s in range(self._get_u32(shape_vec)):
                    shape.append(self._get_i32(shape_vec + 4 + s * 4))

            buf_field = self._get_field_offset(t_table, 2)
            buf_idx = self._get_u32(buf_field) if buf_field else 0

            name_field = self._get_field_offset(t_table, 3)
            name = self._get_string(name_field) if name_field else f"tensor_{i}"

            self._tensor_meta.append({
                'name': name,
                'index': i,
                'shape': np.array(shape, dtype=np.int32),
                'dtype': np.float32,
                'buffer_idx': buf_idx
            })

            buf_data = buffers[buf_idx] if buf_idx < len(buffers) else b''
            if buf_data:
                arr = np.frombuffer(buf_data, dtype=np.float32).reshape(shape)
                self._tensors[i] = arr.copy()
            else:
                self._tensors[i] = None

        # SubGraph inputs (field 1)
        inputs_field = self._get_field_offset(sg_table, 1)
        inputs_vec = inputs_field + self._get_u32(inputs_field)
        self._input_indices = [self._get_i32(inputs_vec + 4 + k * 4) for k in range(self._get_u32(inputs_vec))]

        # SubGraph outputs (field 2)
        outputs_field = self._get_field_offset(sg_table, 2)
        outputs_vec = outputs_field + self._get_u32(outputs_field)
        self._output_indices = [self._get_i32(outputs_vec + 4 + k * 4) for k in range(self._get_u32(outputs_vec))]

        # SubGraph operators (field 3)
        ops_field = self._get_field_offset(sg_table, 3)
        ops_vec = ops_field + self._get_u32(ops_field)
        num_ops = self._get_u32(ops_vec)

        for i in range(num_ops):
            op_elem = ops_vec + 4 + i * 4
            op_table = op_elem + self._get_u32(op_elem)

            op_idx_field = self._get_field_offset(op_table, 0)
            op_code_idx = struct.unpack_from('<I', self._buf, op_idx_field)[0] if op_idx_field else 0
            opcode = opcodes[op_code_idx]

            in_field = self._get_field_offset(op_table, 1)
            in_vec = in_field + self._get_u32(in_field)
            in_tensors = [self._get_i32(in_vec + 4 + k * 4) for k in range(self._get_u32(in_vec))]

            out_field = self._get_field_offset(op_table, 2)
            out_vec = out_field + self._get_u32(out_field)
            out_tensors = [self._get_i32(out_vec + 4 + k * 4) for k in range(self._get_u32(out_vec))]

            opt_field = self._get_field_offset(op_table, 4)
            fused_activation = 0
            if opt_field:
                opt_table = opt_field + self._get_u32(opt_field)
                act_field = self._get_field_offset(opt_table, 0)
                if act_field:
                    fused_activation = struct.unpack_from('<b', self._buf, act_field)[0]

            self._ops.append({
                'opcode': opcode,
                'inputs': in_tensors,
                'outputs': out_tensors,
                'fused_activation': fused_activation
            })

    def allocate_tensors(self):
        """No-op for API compatibility with tf.lite.Interpreter."""
        pass

    def get_input_details(self):
        return [self._tensor_meta[i] for i in self._input_indices]

    def get_output_details(self):
        return [self._tensor_meta[i] for i in self._output_indices]

    def set_tensor(self, tensor_index, value):
        self._tensors[tensor_index] = np.array(value, dtype=np.float32)

    def get_tensor(self, tensor_index):
        return self._tensors[tensor_index]

    def invoke(self):
        for op in self._ops:
            code = op['opcode']
            in_indices = op['inputs']
            out_indices = op['outputs']

            # FULLY_CONNECTED = 9
            if code == 9:
                x = self._tensors[in_indices[0]]
                w = self._tensors[in_indices[1]]  # shape: (out_features, in_features)
                b = self._tensors[in_indices[2]] if len(in_indices) > 2 and in_indices[2] >= 0 else None

                out = np.matmul(x, w.T)
                if b is not None:
                    out = out + b

                act = op['fused_activation']
                if act == 1:  # RELU
                    out = np.maximum(0.0, out)
                elif act == 2:  # RELU_N1_TO_1
                    out = np.clip(out, -1.0, 1.0)
                elif act == 3:  # RELU6
                    out = np.clip(out, 0.0, 6.0)
                elif act == 4:  # TANH
                    out = np.tanh(out)

                self._tensors[out_indices[0]] = out

            # SOFTMAX = 25
            elif code == 25:
                x = self._tensors[in_indices[0]]
                shift_x = x - np.max(x, axis=-1, keepdims=True)
                exps = np.exp(shift_x)
                out = exps / np.sum(exps, axis=-1, keepdims=True)
                self._tensors[out_indices[0]] = out

            # RELU = 19
            elif code == 19:
                x = self._tensors[in_indices[0]]
                self._tensors[out_indices[0]] = np.maximum(0.0, x)

            # RESHAPE = 22
            elif code == 22:
                x = self._tensors[in_indices[0]]
                shape = self._tensors[in_indices[1]].astype(int)
                self._tensors[out_indices[0]] = np.reshape(x, shape)

            else:
                raise NotImplementedError(f"Unsupported TFLite opcode: {code}")


def Interpreter(model_path=None, model_content=None, num_threads=1):
    """
    Factory function returning a TensorFlow Lite Interpreter.
    Uses official tflite_runtime / ai_edge_litert / tensorflow.lite if available,
    otherwise uses the built-in NumPyTFLiteInterpreter.
    """
    if _TFLiteInterpreter is not None:
        try:
            return _TFLiteInterpreter(model_path=model_path, num_threads=num_threads)
        except Exception:
            pass
    return NumPyTFLiteInterpreter(
        model_path=model_path,
        model_content=model_content,
        num_threads=num_threads
    )
