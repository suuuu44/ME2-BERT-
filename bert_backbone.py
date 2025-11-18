import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class BertBackbone(nn.Module):
    """
    BERT 编码器 Backbone 模块
    ------------------------------------------------------------
    作用:
      - 将 token ids → 语义嵌入 (sequence_output, pooled_output)
      - 可控制冻结前若干层 BERT 参数
      - 可选择是否使用 BERT 自带的 pooler 输出
    """

    def __init__(self,
                 model_name: str = "bert-base-uncased",
                 output_hidden_states: bool = False,
                 freeze_layers: int = 0,
                 use_pooler_output: bool = False,
                 device: str = "cpu"):
        super().__init__()
        self.device = device
        self.use_pooler_output = use_pooler_output
        self.config = AutoConfig.from_pretrained(
            model_name,
            output_hidden_states=output_hidden_states
        )
        self.bert = AutoModel.from_pretrained(model_name, config=self.config)

        # ======== 冻结指定层 ========
        if freeze_layers > 0:
            for name, param in self.bert.named_parameters():
                if name.startswith("embeddings"):
                    param.requires_grad = False
                else:
                    for layer_idx in range(freeze_layers):
                        if name.startswith(f"encoder.layer.{layer_idx}."):
                            param.requires_grad = False
                            break  # 减少多余遍历

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state  # [B, T, H]
        pooled_output = outputs.pooler_output  # [B, H]
        return last_hidden_state, pooled_output



