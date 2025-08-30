import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput
from transformers.modeling_outputs import Seq2SeqLMOutput


class T5ForPretrain(T5ForConditionalGeneration):
    def __init__(self, t5_model, args):
        super().__init__(t5_model.config)
        self.args = args
        self.config = t5_model.config
        self.shared = t5_model.shared
        self.encoder = t5_model.encoder
        self.decoder = t5_model.decoder

        if self.args.use_origin_head == "True":
            self.lm_head = t5_model.lm_head
            print("Using original LM head")
        else:
            self.lm_head = nn.Linear(self.config.d_model, self.config.vocab_size, bias=False)

        print("LM head shape:", self.lm_head.weight.shape)

        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs
    ):
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = self._shift_right(labels)

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            head_mask=decoder_head_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        if self.config.tie_word_embeddings:
            sequence_output = sequence_output * (self.model_dim ** -0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss = self.loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))

        if not return_dict:
            return loss

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

class T5ForPretrainDPO(T5ForPretrain):
    def forward(self, input_ids=None,
                attention_mask=None,
                decoder_input_ids=None,
                decoder_attention_mask=None,
                head_mask=None,
                decoder_head_mask=None, 
                cross_attn_head_mask=None, 
                encoder_outputs=None, 
                past_key_values=None, 
                inputs_embeds=None, 
                decoder_inputs_embeds=None, 
                labels=None, use_cache=None, 
                output_attentions=None, 
                output_hidden_states=None, 
                return_dict=True):
        
        return super().forward(input_ids, 
                               attention_mask, 
                               decoder_input_ids, 
                               decoder_attention_mask, 
                               head_mask, 
                               decoder_head_mask, 
                               cross_attn_head_mask, 
                               encoder_outputs, 
                               past_key_values, 
                               inputs_embeds, 
                               decoder_inputs_embeds, 
                               labels, use_cache, 
                               output_attentions, 
                               output_hidden_states, 
                               return_dict)