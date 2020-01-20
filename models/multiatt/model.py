"""
Let's get the relationships yo
"""
from typing import Dict, List, Any, Callable, Optional

import torch
import torch.nn.functional as F
import torch.nn.parallel
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import TextFieldEmbedder, Seq2SeqEncoder, FeedForward, InputVariationalDropout,TimeDistributed,SimilarityFunction
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.modules.matrix_attention import BilinearMatrixAttention, DotProductMatrixAttention
from utils.detector import SimpleDetector
from allennlp.nn.util import masked_softmax, weighted_sum, replace_masked_values
from allennlp.nn import InitializerApplicator
from allennlp.modules import FeedForward, MatrixAttention

from models.multiatt.LSTM_model import MYLSTM
from models.multiatt.LSTM_source import source_LSTM

import logging
import numpy as np
from torch.nn.modules import BatchNorm1d


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Model.register("MultiHopAttentionQA")
class AttentionQA(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 span_encoder: Seq2SeqEncoder,
                 reasoning_encoder: Seq2SeqEncoder,
                 input_dropout: float = 0.1,
                 hidden_dim_maxpool: int = 512,
                 class_embs: bool=True,
                 reasoning_use_obj: bool=True,
                 reasoning_use_answer: bool=True,
                 reasoning_use_question: bool=True,
                 pool_reasoning: bool = True,
                 pool_answer: bool = True,
                 pool_question: bool = False,
                 preload_path: str = "source_model.th",
                 initializer: InitializerApplicator = InitializerApplicator(),
                 ):
        super(AttentionQA, self).__init__(vocab)

        self.detector = SimpleDetector(pretrained=True, average_pool=True, semantic=class_embs, final_dim=512)
        ###################################################################################################

        self.rnn_input_dropout = TimeDistributed(InputVariationalDropout(input_dropout)) if input_dropout > 0 else None

        self.span_encoder = TimeDistributed(span_encoder)
        self.reasoning_encoder = TimeDistributed(reasoning_encoder)
        self.BiLSTM = TimeDistributed(MYLSTM(1280, 512, 256))
        self.source_encoder = TimeDistributed(source_LSTM(768, 256))


        self.span_attention = BilinearMatrixAttention(
            matrix_1_dim=span_encoder.get_output_dim(),
            matrix_2_dim=span_encoder.get_output_dim(),
        )
        self.span_attention_2 = BilinearMatrixAttention(
            matrix_1_dim=span_encoder.get_output_dim(),
            matrix_2_dim=span_encoder.get_output_dim(),
        )

        self.obj_attention = BilinearMatrixAttention(
            matrix_1_dim=span_encoder.get_output_dim(),
            matrix_2_dim=self.detector.final_dim,
        )

        self.obj_attention_2 = BilinearMatrixAttention(
            matrix_1_dim=span_encoder.get_output_dim(),
            matrix_2_dim=self.detector.final_dim,
        )

        self._matrix_attention = DotProductMatrixAttention()
        #self._matrix_attention = MatrixAttention(similarity_function)

        self.reasoning_use_obj = reasoning_use_obj
        self.reasoning_use_answer = reasoning_use_answer
        self.reasoning_use_question = reasoning_use_question
        self.pool_reasoning = pool_reasoning
        self.pool_answer = pool_answer
        self.pool_question = pool_question
        dim = sum([d for d, to_pool in [(reasoning_encoder.get_output_dim(), self.pool_reasoning),
                                        (span_encoder.get_output_dim(), self.pool_answer),
                                        (span_encoder.get_output_dim(), self.pool_question)] if to_pool])

        self.final_mlp = torch.nn.Sequential(
            torch.nn.Dropout(input_dropout, inplace=False),
            torch.nn.Linear(dim, hidden_dim_maxpool),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(input_dropout, inplace=False),
            torch.nn.Linear(hidden_dim_maxpool, 1),
        )
        self.final_mlp_2 = torch.nn.Sequential(
            torch.nn.Dropout(input_dropout, inplace=False),
            torch.nn.Linear(dim, hidden_dim_maxpool),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(input_dropout, inplace=False),
            torch.nn.Linear(hidden_dim_maxpool, 1),
        )

        self.answer_BN = torch.nn.Sequential(
            BatchNorm1d(512)
        )
        self.question_BN = torch.nn.Sequential(
            BatchNorm1d(512)
        )
        self.source_answer_BN = torch.nn.Sequential(
            BatchNorm1d(512)
        )
        self.source_question_BN = torch.nn.Sequential(
            BatchNorm1d(512)
        )
        self.image_BN = BatchNorm1d(512)
        self.final_BN = torch.nn.Sequential(
            BatchNorm1d(512)
        )
        self.final_mlp_linear = torch.nn.Sequential(
            torch.nn.Linear(512,1)
        )
        self.final_mlp_pool = torch.nn.Sequential(
            torch.nn.Linear(2560,512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(input_dropout,inplace=False),
        )


        self._accuracy = CategoricalAccuracy()
        self._loss = torch.nn.CrossEntropyLoss()
        initializer(self)

        if preload_path is not None:
            logger.info("Preloading!")
            preload = torch.load(preload_path)
            own_state = self.state_dict()
            for name, param in preload.items():
                #if name[0:8] == "_encoder":
                #    suffix = "._module."+name[9:]
                #    logger.info("preload paramter {}".format("span_encoder"+suffix))
                #    own_state["span_encoder"+suffix].copy_(param)
                #新引入的source_encoder
                if name[0:4] == "LSTM":
                    suffix = "._module" + name[4:]
                    logger.info("preload paramter {}".format("source_encoder"+suffix))
                    own_state["source_encoder"+suffix].copy_(param)

    def _collect_obj_reps(self, span_tags, object_reps):
        """
        Collect span-level object representations
        :param span_tags: [batch_size, ..leading_dims.., L]
        :param object_reps: [batch_size, max_num_objs_per_batch, obj_dim]
        :return:
        """
        span_tags_fixed = torch.clamp(span_tags, min=0)  # In case there were masked values here
        row_id = span_tags_fixed.new_zeros(span_tags_fixed.shape)
        row_id_broadcaster = torch.arange(0, row_id.shape[0], step=1, device=row_id.device)[:, None]

        # Add extra diminsions to the row broadcaster so it matches row_id
        leading_dims = len(span_tags.shape) - 2
        for i in range(leading_dims):
            row_id_broadcaster = row_id_broadcaster[..., None]
        row_id += row_id_broadcaster
        return object_reps[row_id.view(-1), span_tags_fixed.view(-1)].view(*span_tags_fixed.shape, -1)

    def embed_span(self, span, span_tags, span_mask, object_reps):
        """
        :param span: Thing that will get embed and turned into [batch_size, ..leading_dims.., L, word_dim]
        :param span_tags: [batch_size, ..leading_dims.., L]
        :param object_reps: [batch_size, max_num_objs_per_batch, obj_dim]
        :param span_mask: [batch_size, ..leading_dims.., span_mask
        :return:
        """
        retrieved_feats = self._collect_obj_reps(span_tags, object_reps)

        span_rep = span['bert']        

        # add recurrent dropout here
        if self.rnn_input_dropout:
            span_rep = self.rnn_input_dropout(span_rep)

        source_knowledge = self.source_encoder(span_rep,span_mask)

        span_rep = torch.cat((span_rep, retrieved_feats ), -1)
        if self.rnn_input_dropout:
            span_rep = self.rnn_input_dropout(span_rep)
            source_knowledge_rep = self.rnn_input_dropout(source_knowledge)
   
        return self.BiLSTM(span_rep,source_knowledge_rep,span_mask), retrieved_feats, source_knowledge


    def _last_dimension_applicator(self,function_to_apply: Callable[[torch.Tensor, Optional[torch.Tensor]], torch.Tensor],
                               tensor: torch.Tensor,
                               mask: Optional[torch.Tensor] = None):
        """
        Takes a tensor with 3 or more dimensions and applies a function over the last dimension.  We
        assume the tensor has shape ``(batch_size, ..., sequence_length)`` and that the mask (if given)
        has shape ``(batch_size, sequence_length)``.  We first unsqueeze and expand the mask so that it
        has the same shape as the tensor, then flatten them both to be 2D, pass them through
        the function and put the tensor back in its original shape.
        """
        tensor_shape = tensor.size()
        reshaped_tensor = tensor.view(-1, tensor.size()[-1])
        if mask is not None:
            while mask.dim() < tensor.dim():
                mask = mask.unsqueeze(1)
            mask = mask.expand_as(tensor).contiguous().float()
            mask = mask.view(-1, mask.size()[-1])
        reshaped_result = function_to_apply(reshaped_tensor, mask)
        return reshaped_result.view(*tensor_shape)


    def last_dim_softmax(self, tensor: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Takes a tensor with 3 or more dimensions and does a masked softmax over the last dimension.  We
        assume the tensor has shape ``(batch_size, ..., sequence_length)`` and that the mask (if given)
        has shape ``(batch_size, sequence_length)``.
        """
        return self._last_dimension_applicator(masked_softmax, tensor, mask)


    def forward(self,
                images: torch.Tensor,
                objects: torch.LongTensor,
                segms: torch.Tensor,
                boxes: torch.Tensor,
                box_mask: torch.LongTensor,
                question: Dict[str, torch.Tensor],
                question_tags: torch.LongTensor,
                question_mask: torch.LongTensor,
                answers: Dict[str, torch.Tensor],
                answer_tags: torch.LongTensor,
                answer_mask: torch.LongTensor,
                metadata: List[Dict[str, Any]] = None,
                label: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
        """
        :param images: [batch_size, 3, im_height, im_width]
        :param objects: [batch_size, max_num_objects] Padded objects
        :param boxes:  [batch_size, max_num_objects, 4] Padded boxes
        :param box_mask: [batch_size, max_num_objects] Mask for whether or not each box is OK
        :param question: AllenNLP representation of the question. [batch_size, num_answers, seq_length]
        :param question_tags: A detection label for each item in the Q [batch_size, num_answers, seq_length]
        :param question_mask: Mask for the Q [batch_size, num_answers, seq_length]
        :param answers: AllenNLP representation of the answer. [batch_size, num_answers, seq_length]
        :param answer_tags: A detection label for each item in the A [batch_size, num_answers, seq_length]
        :param answer_mask: Mask for the As [batch_size, num_answers, seq_length]
        #:param ind: Ignore, this is about which dataset item we're on
        :param metadata: Ignore, this is about which dataset item we're on
        :param label: Optional, which item is valid
        :return: shit
        """
        # Trim off boxes that are too long. this is an issue b/c dataparallel, it'll pad more zeros that are
        # not needed
        max_len = int(box_mask.sum(1).max().item())
        objects = objects[:, :max_len]
        box_mask = box_mask[:, :max_len]
        boxes = boxes[:, :max_len]
        segms = segms[:, :max_len]

        for tag_type, the_tags in (('question', question_tags), ('answer', answer_tags)):
            if int(the_tags.max()) > max_len:
                raise ValueError("Oh no! {}_tags has maximum of {} but objects is of dim {}. Values are\n{}".format(
                    tag_type, int(the_tags.max()), objects.shape, the_tags
                ))

        obj_reps = self.detector(images=images, boxes=boxes, box_mask=box_mask, classes=objects, segms=segms)


        # Now get the question/answer representations
        q_rep, q_obj_reps, q_source = self.embed_span(question, question_tags, question_mask, obj_reps['obj_reps'])
        a_rep, a_obj_reps, a_source = self.embed_span(answers, answer_tags, answer_mask, obj_reps['obj_reps'])

        batch_size,num_options,q_pad_len,_ = q_rep.shape
        batch_size,num_options,a_pad_len,_ = a_rep.shape

        a_p = a_rep[:,:,-1,:]
        q_p = q_rep[:,:,-1,:]
        a_source_p = a_source[:,:,-1,:]
        q_source_p = q_source[:,:,-1,:]
  
        a_p = a_p.view(batch_size*num_options,-1)
        a_p = self.answer_BN(a_p)
        a_p = a_p.view(batch_size,num_options,-1)

        q_p = q_p.view(batch_size*num_options,-1)
        q_p = self.question_BN(q_p)
        q_p = q_p.view(batch_size,num_options,-1)
 
        a_source_p = a_source_p.view(batch_size*num_options,-1)
        a_source_p = self.source_answer_BN(a_source_p)
        a_source_p = a_source_p.view(batch_size,num_options,-1)

        q_source_p = q_source_p.view(batch_size*num_options,-1)
        q_source_p = self.source_question_BN(q_source_p)
        q_source_p = q_source_p.view(batch_size,num_options,-1)

        # image part
        assert (obj_reps['obj_reps'][:,0,:].shape == (batch_size, 512))
        img_global = obj_reps['obj_reps'][:,0,:] #  the background i.e. whole image
        img_global = self.image_BN(img_global)
        img_global = img_global[:,None,:]
        img_global = img_global.repeat(1,4,1) # (batch_size, 4, 512)
        assert (img_global.shape == (batch_size,num_options,512))


        # Perform global reasoning cues composition
        QAI_g = torch.cat((a_p,q_p,a_source_p,q_source_p,img_global),-1)

        QAI_g = self.final_mlp_pool(QAI_g)
        QAI_g = QAI_g.view(batch_size*num_options,-1)
        QAI_g = self.final_BN(QAI_g)
        QAI_g =QAI_g.view(batch_size,num_options,-1)
        logits_g = self.final_mlp_linear(QAI_g).squeeze(2)


        ################################################################
        # Perform Q by A attention
        # [batch_size, 4, question_length, answer_length]
        qa_similarity = self.span_attention(
            q_rep.view(batch_size * num_options, q_pad_len, -1),
            a_rep.view(batch_size * num_options, a_pad_len, -1),
        ).view(batch_size, num_options, q_pad_len, a_pad_len)
        qa_attention_weights = masked_softmax(qa_similarity, question_mask[..., None], dim=2)
        attended_q = torch.einsum('bnqa,bnqd->bnad', (qa_attention_weights, q_rep))

        #Perform A by Q attention
        # [batch_size, 4, answer_length, question_length]
        aq_similarity = self.span_attention_2(
            a_rep.view(batch_size * num_options, a_pad_len, -1),
            q_rep.view(batch_size * num_options, q_pad_len, -1),
        ).view(batch_size, num_options, a_pad_len, q_pad_len)
        aq_attention_weights = masked_softmax(aq_similarity, answer_mask[..., None], dim=2)
        attended_a = torch.einsum('bnaq,bnad->bnqd', (aq_attention_weights, a_rep))


        #Perform attention transfer
        qa_similarity_source = self._matrix_attention(
            q_source.view(batch_size * num_options, q_pad_len, -1),
            a_source.view(batch_size * num_options, a_pad_len, -1)).view(batch_size, num_options, q_pad_len, a_pad_len)

        aq_similarity_source = self._matrix_attention(
            a_source.view(batch_size * num_options, a_pad_len, -1),
            q_source.view(batch_size * num_options, q_pad_len, -1)).view(batch_size, num_options, a_pad_len, q_pad_len)

        attention_loss = torch.nn.MSELoss()(qa_similarity,qa_similarity_source) +  torch.nn.MSELoss()(aq_similarity,aq_similarity_source)


        ################################################################
        # Have a second attention over the objects, do A by Objs
        # [batch_size, 4, answer_length, num_objs]
        atoo_similarity = self.obj_attention(a_rep.view(batch_size, num_options * a_pad_len, -1),
                                             obj_reps['obj_reps']).view(batch_size, num_options,
                                                            a_pad_len, obj_reps['obj_reps'].shape[1])
        atoo_attention_weights = masked_softmax(atoo_similarity, box_mask[:,None,None])
        attended_o = torch.einsum('bnao,bod->bnad', (atoo_attention_weights, obj_reps['obj_reps']))


        # Have a second attention over the objects, do obj by Q
        # [batch_size, 4, answer_length, num_objs]
        otoq_similarity = self.obj_attention_2(q_rep.view(batch_size, num_options * q_pad_len, -1),
                                             obj_reps['obj_reps']).view(batch_size, num_options,
                                                            q_pad_len, obj_reps['obj_reps'].shape[1])
        otoq_attention_weights = masked_softmax(otoq_similarity, box_mask[:,None,None])
        attended_o_2 = torch.einsum('bnqo,bod->bnqd', (otoq_attention_weights, obj_reps['obj_reps']))
 

        #################################################################
        #reasoning composition part
        reasoning_inp = torch.cat([x for x, to_pool in [(a_rep, self.reasoning_use_answer),
                                                           (attended_o, self.reasoning_use_obj),
                                                           (attended_q, self.reasoning_use_question)]
                                      if to_pool], -1)

        reasoning_inp_2 = torch.cat([x for x, to_pool in [(q_rep, self.reasoning_use_question),
                                                           (attended_o_2, self.reasoning_use_obj),
                                                           (attended_a, self.reasoning_use_answer)]
                                      if to_pool], -1)


        if self.rnn_input_dropout is not None:
            reasoning_inp = self.rnn_input_dropout(reasoning_inp)
        reasoning_output = self.reasoning_encoder(reasoning_inp, answer_mask)

        if self.rnn_input_dropout is not None:
            reasoning_inp_2 = self.rnn_input_dropout(reasoning_inp_2)
        reasoning_output_2 = self.reasoning_encoder(reasoning_inp_2, question_mask)

        ##################logits_1   
        things_to_pool = torch.cat([x for x, to_pool in [(reasoning_output, self.pool_reasoning),
                                                         (a_rep, self.pool_answer),
                                                         (attended_q, self.pool_question)] if to_pool], -1)
        pooled_rep = replace_masked_values(things_to_pool,answer_mask[...,None], -1e7).max(2)[0]
        logits_1 = self.final_mlp(pooled_rep).squeeze(2)


        ##################logits_2
        things_to_pool_2 = torch.cat([x for x, to_pool in [(reasoning_output_2, self.pool_reasoning),
                                                         (attended_a, self.pool_answer),
                                                         (q_rep, self.pool_question)] if to_pool], -1)
        pooled_rep_2 = replace_masked_values(things_to_pool_2,question_mask[...,None], -1e7).max(2)[0]
        logits_2 = self.final_mlp_2(pooled_rep_2).squeeze(2)

        
        lambda_f = 0.7
        lambda_g = 0.3

        logits_f = logits_1 + logits_2
        logits_g = logits_g


        logits = lambda_f*logits_f + lambda_g*logits_g

        ###########################################

        accf = torch.mean((label == logits_f.argmax(1)).float())
        accg = torch.mean((label == logits_g.argmax(1)).float())
        accuracy = torch.mean((label == logits.argmax(1)).float())

        prob_f = F.softmax(logits_f,dim=-1)
        prob_g = F.softmax(logits_g,dim=-1)
        class_probabilities = lambda_f*prob_f + lambda_g*prob_g

        output_dict = {"label_logits": logits, "label_probs": class_probabilities,
                       'cnn_regularization_loss': obj_reps['cnn_regularization_loss'],
                       'accuracy':accuracy, 'accf': accf, 'accg': accg,
                       'attention_loss': attention_loss,
                       }
        if label is not None:
            loss = lambda_f*self._loss(logits_f, label.long().view(-1)) + lambda_g*self._loss(logits_g, label.long().view(-1)) + 100*attention_loss
            output_dict["loss"] = loss[None]

        return output_dict
