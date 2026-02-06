from calendar import c
from httpx import get
import torch
import torch.nn as nn
import torch.nn.functional as F

def continual_orthogonal_loss(model):
    ########################### Regularization ##########################
    orthogonal_loss = 0.
    lambad_1 = 0.5
    lambad_2 = 0
    for name, param in model.named_parameters():
        if "lora_A" in name:
            for name_, param_ in model.named_parameters():
                if "loranew_A" in name_ and name.split("lora_A")[0] == name_.split("loranew_A")[0]:
                    orthogonal_loss += torch.abs(torch.mm(param, param_.T)).sum() # [r * dim] * [dim * r]
                    break # target modules have been matched

    # l2-normalization for loranew_A/B
    l2_loss = 0.
    for name, param in model.named_parameters():
        if "loranew_" in name:
            l2_loss += torch.norm(param, p=2)
    
    return lambad_1 * orthogonal_loss + lambad_2 * l2_loss

# def orthogonal_loss(cur_modal, model):
#     ########################### Regularization ##########################
#     orthogonal_loss = 0.
#     lambad_1 = 1.0

#     param_dict = {name: param for name, param in model.named_parameters()}
    
#     for name, param in param_dict.items():
#         if "lora_A" in name:
#             param_A = param
            
#             param_B = param_dict[name.replace("lora_A", "lora_B")]
            
#             lora_refer_A = param_dict[name.replace("lora_A.weight", "lora_refer_A")]
            
#             lora_refer_B = param_dict[name.replace("lora_A.weight", "lora_refer_B")]
            
#             weight = param_dict[name.replace("lora_A.", "")]
            
#             base_ref = lora_refer_A @ lora_refer_B
#             loss = torch.pow(weight - base_ref, 2).mean()  # 保证分解矩阵和模型原始权重一样。
#             loss += torch.abs(torch.mm(param_A, lora_refer_B.T)).sum()  # A正交
#             loss += torch.abs(torch.mm(param_B, lora_refer_A.T)).sum()  # B正交

#             orthogonal_loss += loss

#     return {f'orthogonal_loss_{cur_modal}': lambad_1 * orthogonal_loss}

class MultiModalUncertaintyWeightingStrategy(nn.Module):
    def __init__(self, args):
        super(MultiModalUncertaintyWeightingStrategy, self).__init__()

        self.tasks = args.train_modal_list
        self.tasks_len = len(args.train_modal_list)
        self.balancer = nn.ModuleDict()

        # self.balancer_weight =nn.ParameterDict()

        for m in self.tasks:
            if args.task_balancer == "uncertainty":
                task_len = 0
                
                if args.use_orthogonal_loss:
                    task_len += 1
                
                if m == "rgbd":
                    if args.cross_align:
                        task_len += 2 * (self.tasks_len - 1)
                    if args.uni_align:
                        task_len += 2
                elif m == "point":
                    if args.cross_align:
                        task_len += 1 * (self.tasks_len - 1)
                    if args.uni_align:
                        task_len += 5
                    if args.multi_modal_distill and 'point'in args.multi_modal_distill_modal_list:
                        task_len += len(args.pc_logits_path_list)
                else:
                    task_len += 1
                    if args.cross_align:
                        task_len += 1 * (self.tasks_len - 1)
                    if args.uni_align:
                        task_len += 2

                self.balancer[m] = UncertaintyWeightingStrategy(task_len)
            else:
                self.balancer[m] = NoWeightingStrategy()
            # self.balancer_weight[m]= nn.Parameter(torch.ones([]))

    def forward(self, cur_modal, task_losses):
        weighted_task_losses = self.balancer[cur_modal](task_losses)

        total_loss = sum(weighted_task_losses.values())

        return weighted_task_losses, total_loss


class UncertaintyWeightingStrategy(nn.Module):
    """Uncertainty weighting strategy"""

    def __init__(self, tasks):
        super(UncertaintyWeightingStrategy, self).__init__()

        self.tasks = tasks
        self.log_vars = nn.Parameter(torch.zeros(tasks))

    def forward(self, task_losses):
        losses_tensor = torch.stack(list(task_losses.values()))
        non_zero_losses_mask = losses_tensor != 0.0

        # calculate weighted losses
        losses_tensor = torch.exp(-self.log_vars) * losses_tensor + self.log_vars

        # if some loss was 0 (i.e. task was dropped), weighted loss should also be 0 and not just log_var as no information was gained
        losses_tensor *= non_zero_losses_mask

        # return dictionary of weighted task losses
        weighted_task_losses = task_losses.copy()
        weighted_task_losses.update(zip(weighted_task_losses, losses_tensor))
        return weighted_task_losses


class NoWeightingStrategy(nn.Module):
    """No weighting strategy"""

    def __init__(self, **kwargs):
        super(NoWeightingStrategy, self).__init__()

    def forward(self, task_losses):
        return task_losses


def kd_normalize(logit):
    mean = logit.mean(dim=-1, keepdims=True)
    stdv = logit.std(dim=-1, keepdims=True)
    return (logit - mean) / (1e-7 + stdv)


def KD_Norm_Loss(logits_student_in, logits_teacher_in, temperature, logit_stand):
    logits_student = (
        kd_normalize(logits_student_in) if logit_stand else logits_student_in
    )
    logits_teacher = (
        kd_normalize(logits_teacher_in) if logit_stand else logits_teacher_in
    )
    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
    loss_kd *= temperature**2
    return loss_kd


def multi_cross_modal_kd_loss(
    modal_list,
    samples_proj2text_features,
    samples_text_features,
    modal_labels_features,
    args,
    logit_scale,
):
    cross_modal_kd_loss = {}

    for m_source in modal_list:
        loss = single_cross_modal_kd_loss(
            m_source,
            modal_list,
            samples_proj2text_features,
            samples_text_features,
            modal_labels_features,
            args,
            logit_scale,
        )

        cross_modal_kd_loss.update(**loss)

    return cross_modal_kd_loss


def single_cross_modal_kd_loss(
    cur_modal,
    modal_list,
    samples_proj2text_features,
    samples_text_features,
    modal_labels_features,
    args,
    logit_scale,
):
    T = args.temperature

    dict_kd_loss = {}

    for m_target in modal_list:
        for anchor, _ in samples_proj2text_features[cur_modal].items():
            if m_target != cur_modal:
                sim_text = (
                    logit_scale[cur_modal]
                    * samples_text_features[cur_modal]
                    @ modal_labels_features[m_target].T
                )

                sim_s2t = (
                    logit_scale[cur_modal]
                    * samples_proj2text_features[cur_modal][anchor]
                    @ modal_labels_features[m_target].T
                )

                kd_loss = KD_Norm_Loss(sim_s2t, sim_text, T, True)

                if anchor == cur_modal:
                    dict_kd_loss.update(
                        {f"cross_kd_loss_{cur_modal}2{m_target}": kd_loss}
                    )
                else:
                    dict_kd_loss.update(
                        {f"cross_kd_loss_{cur_modal}_{anchor}2{m_target}": kd_loss}
                    )

    return dict_kd_loss


def single_uni_modal_nce_loss(
    cur_modal,
    samples_proj2text_features,
    samples_text_features,
    logit_scale,
    epoch=0,
):
    T = 4
    dict_nce_loss = {}

    device = samples_text_features[cur_modal].device

    nce_labels = torch.arange(samples_text_features[cur_modal].shape[0]).to(device)

    sim_dict = {}

    for anchor, _ in samples_proj2text_features[cur_modal].items():
        
        # if anchor !='image':
        
        sim_anchor2text = (
            logit_scale[anchor]
            * samples_proj2text_features[cur_modal][anchor]
            @ samples_text_features[cur_modal].T
        )

        sim_dict[anchor] = sim_anchor2text

        nce_anchor2text_loss = (
            F.cross_entropy(sim_anchor2text, nce_labels)
            + F.cross_entropy(sim_anchor2text.T, nce_labels)
        ) / 2

        dict_nce_loss.update({f"nce_loss_{cur_modal}_{anchor}2t": nce_anchor2text_loss})

        for target, _ in samples_proj2text_features[cur_modal].items():
            if anchor != target and anchor != cur_modal:
                sim_anchor2target = (
                    logit_scale[anchor]
                    * samples_proj2text_features[cur_modal][anchor]
                    @ samples_proj2text_features[cur_modal][target].T
                )

                nce_anchor2target_loss = (
                    F.cross_entropy(sim_anchor2target, nce_labels)
                    + F.cross_entropy(sim_anchor2target.T, nce_labels)
                ) / 2

                dict_nce_loss.update(
                    {f"nce_loss_{cur_modal}_{anchor}2{target}": nce_anchor2target_loss}
                )

    if len(sim_dict) > 1:
        kd_loss=0
        for k, v in sim_dict.items():
            if k != 'image':
                kd_loss += KD_Norm_Loss(v, sim_dict['image'], T, True)
        dict_nce_loss.update({f"uni_kd_loss_{cur_modal}": kd_loss})

    return dict_nce_loss


def multi_uni_modal_nce_loss(
    modal_list, samples_proj2text_features, samples_text_features, logit_scale, epoch=0
):
    dict_nce_loss = {}
    for modal in modal_list:
        loss = single_uni_modal_nce_loss(
            modal,
            samples_proj2text_features,
            samples_text_features,
            logit_scale,
            epoch,
        )
        dict_nce_loss.update(**loss)
    return dict_nce_loss


def cross_modal_cls(
    cur_modal,
    samples_proj2text_features,
    modal_labels_features,
    targets,
    logit_scale,
):
    if cur_modal == "audio":
        targets = targets.argmax(dim=1)

    
    sim_m2l = (
        logit_scale[cur_modal]
        * samples_proj2text_features[cur_modal][cur_modal]
        @ modal_labels_features[cur_modal].T
    )
    
    m2t_cls_loss = F.cross_entropy(sim_m2l, targets)

    dict_cm_cls_loss = {f"cm_cls_loss_{cur_modal}": m2t_cls_loss}

    m2t_cls_acc = (
        (sim_m2l.detach().softmax(dim=-1).argmax(dim=1) == targets).float().mean()
    )

    dict_cm_cls_acc = {f"cm_cls_acc_{cur_modal}": m2t_cls_acc}

    return dict_cm_cls_loss, dict_cm_cls_acc


def multi_cross_modal_cls(
    modal_list,
    samples_proj2text_features,
    modal_labels_features,
    targets,
    logit_scale,
    accum_iter=0,
):
    dict_cm_cls_loss = {}
    dict_cm_cls_acc = {}

    for modal in modal_list:
        loss, acc = cross_modal_cls(
            modal,
            samples_proj2text_features,
            modal_labels_features,
            targets[modal][accum_iter],
            logit_scale,
        )

        dict_cm_cls_loss.update(**loss)
        dict_cm_cls_acc.update(**acc)

    return dict_cm_cls_loss, dict_cm_cls_acc


def multi_anchor_kd_loss(
    modal_list,
    samples_proj2text_features,
    samples_text_features,
    args,
    logit_scale,
):
    cross_modal_kd_loss = {}

    for m_source in modal_list:
        loss = single_anchor_kd_loss(
            m_source,
            modal_list,
            samples_proj2text_features,
            samples_text_features,
            args,
            logit_scale,
        )

        cross_modal_kd_loss.update(**loss)

    return cross_modal_kd_loss


def single_anchor_kd_loss(
    cur_modal,
    modal_list,
    samples_proj2text_features,
    samples_text_features,
    args,
    logit_scale,
):
    T = args.temperature

    dict_kd_loss = {}

    for m_target in modal_list:
        if m_target != cur_modal:
            sim_text = (
                logit_scale[cur_modal]
                * samples_text_features[cur_modal]
                @ samples_text_features[m_target].T
            )

            if cur_modal == "rgbd":
                sim_d2t = (
                    logit_scale[cur_modal]
                    * samples_proj2text_features[cur_modal]["rgbd"]
                    @ samples_proj2text_features[m_target].T
                )

                kd_loss_d2t_ = KD_Norm_Loss(sim_d2t, sim_text, T, True)

                sim_rgb2t = (
                    logit_scale["image"]
                    * samples_proj2text_features[cur_modal]["image"]
                    @ samples_proj2text_features[m_target].T
                )

                kd_loss_rgb2t = KD_Norm_Loss(sim_rgb2t, sim_text, T, True)

                kd_loss = (kd_loss_d2t_ + kd_loss_rgb2t) / 2

                dict_kd_loss.update({f"anchor_kd_loss_{cur_modal}2{m_target}": kd_loss})

            elif m_target == "rgbd":
                sim_d2t = (
                    logit_scale[cur_modal]
                    * samples_proj2text_features[cur_modal]
                    @ samples_proj2text_features[m_target]["rgbd"].T
                )

                kd_loss_d2t_ = KD_Norm_Loss(sim_d2t, sim_text, T, True)

                sim_rgb2t = (
                    logit_scale[cur_modal]
                    * samples_proj2text_features[cur_modal]
                    @ samples_proj2text_features[m_target]["image"].T
                )

                kd_loss_rgb2t = KD_Norm_Loss(sim_rgb2t, sim_text, T, True)

                kd_loss = (kd_loss_d2t_ + kd_loss_rgb2t) / 2

                dict_kd_loss.update({f"anchor_kd_loss_{cur_modal}2{m_target}": kd_loss})

            else:
                sim_s2t = (
                    logit_scale[cur_modal]
                    * samples_proj2text_features[cur_modal]
                    @ samples_proj2text_features[m_target].T
                )

                kd_loss = KD_Norm_Loss(sim_s2t, sim_text, T, True)

                dict_kd_loss.update({f"anchor_kd_loss_{cur_modal}2{m_target}": kd_loss})

    return dict_kd_loss


def single_kd_clip_loss(
    cur_modal,
    samples_proj2text_features,
    t_samples_proj2text_features,
    args=None
):

    dict_kd_clip_loss = {}
    
    fd_loss = F.mse_loss(samples_proj2text_features[cur_modal], t_samples_proj2text_features[cur_modal])
    
    kd_clip_loss = args.alpha_fd_loss * fd_loss

    dict_kd_clip_loss.update({f"kd_t_loss_{cur_modal}": kd_clip_loss})

    return dict_kd_clip_loss