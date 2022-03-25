import torch
from pathlib import Path
from onnxruntime import (
    GraphOptimizationLevel,
    InferenceSession,
    SessionOptions,
)
from pathlib import Path

class OnnxBase(object):
    def __init__(self, use_gpu: bool = True):
        self.gpu = use_gpu

    def _set_session_option(self):
        options = SessionOptions()
        options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL

        # options.execution_mode = ExecutionMode.ORT_PARALLEL
        # options.execution_mode = ExecutionMode.ORT_PARALLEL
        return options

    def _init_session(self, weight_path: Path):
        session = InferenceSession(
            str(weight_path),
            providers=['CUDAExecutionProvider' if self.gpu else 'CPUExecutionProvider'])

        return session

class OnnxSeq2seq(OnnxBase):
    def __init__(self, encoder_path: str, decoder_path: str, use_gpu: bool = True):
        self.encoder_path = Path(encoder_path)
        self.decoder_path = Path(decoder_path)

        self.gpu = use_gpu

        for weight_path in [self.encoder_path, self.decoder_path]:
            if not weight_path.exists():
                print("OnnxYLV5: File not found: %s" % str(weight_path))
                return

        self.encoder = self._init_session(self.encoder_path)
        self.decoder = self._init_session(self.decoder_path)

    def forward_encoder(self, src):
        encoder_outputs, hidden = self.encoder.run(None, {
            'data_input': src
        })

        return (hidden, encoder_outputs)

    def forward_decoder(self, tgt, memory):
        hidden, encoder_outputs = memory
        decoder_inputs = {
            "tgt": tgt[-1].cpu().numpy(),
            "hidden_input": hidden,
            "encoder_output": encoder_outputs
        }
        
        output, hidden, _ = self.decoder.run(None, decoder_inputs)
        output = torch.from_numpy(output).to(tgt.device).unsqueeze(1)

        return output, (hidden, encoder_outputs)

class OnnxVietOCR(OnnxBase):
    def __init__(self, backbone_path: str, encoder_path: str, decoder_path: str, use_gpu: bool = True):
        self.cnn_path = Path(backbone_path)
        self.transformer = OnnxSeq2seq(encoder_path, decoder_path, use_gpu)

        self.gpu = use_gpu

        if not self.cnn_path.exists():
            print("OnnxYLV5: File not found: %s" % str(self.cnn_path))
            return

        self.cnn_model = self._init_session(self.cnn_path)
        self.input_name = self.cnn_model.get_inputs()[0].name
        
    def cnn(self, img):
        # out = self.cnn.run(None, {self.input_name: img.cpu().numpy()})
        # out = torch.from_numpy(out[0])
        return self.cnn_model.run(None, {self.input_name: img.cpu().numpy()})[0]

    def eval(self):
        # trick
        pass



