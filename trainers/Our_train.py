# import torch

# from loss import Angular_Isotonic_Loss


# def default_train(train_loader, model, optimizer, writer, iter_counter, args):

#     if args.gpu_num > 1:
#         way = model.module.way
#         query_shot = model.module.shots[-1]
#         support_shot = model.module.shots[0]

#     else:
#         way = model.way
#         query_shot = model.shots[-1]
#     target = torch.LongTensor([i // query_shot for i in range(query_shot * way)]).cuda()

#     criterion = Angular_Isotonic_Loss(args.train_way, args.lamda, args.mrg, args.threshold).cuda()

#     lr = optimizer.param_groups[0]['lr']
#     writer.add_scalar('lr', lr, iter_counter)

#     avg_loss = 0
#     avg_acc = 0

#     for i, (inp, _) in enumerate(train_loader):
#         iter_counter += 1

#         if args.gpu_num > 1:
#             inp_spt = inp[:way * support_shot]
#             inp_qry = inp[way * support_shot:]
#             qry_num = inp_qry.shape[0]
#             inp_list = []
#             for i in range(args.gpu_num):
#                 inp_qry_fraction = inp_qry[int(qry_num/i):int(qry_num/(i+1))]
#                 inp_list.append(torch.cat((inp_spt, inp_qry_fraction), dim=0))
#             inp = torch.cat(inp_list, dim=0)

#         inp = inp.cuda()

#         # cos_a1, cos_a2, cos_b1, cos_b2, cos_a2_b2, cos_b2_a2, cos_a4_b4, cos_b4_a4 = model(inp)
#         cos_a1, cos_a2, cos_b1, cos_b2 = model(inp)
#         loss1 = criterion(cos_a1, target)
#         loss2 = criterion(cos_a2, target)
#         loss3 = criterion(cos_b1, target)
#         loss4 = criterion(cos_b2, target)
#         # loss5 = criterion(cos_a2_b2, target)
#         # loss6 = criterion(cos_b2_a2, target)
#         # loss7 = criterion(cos_a4_b4, target)
#         # loss8 = criterion(cos_b4_a4, target)

#         # lossT1 = loss1 + loss2 + loss3 + loss4
#         # lossT2 = loss5 + loss6 + loss7 + loss8
#         # loss = (lossT1 + lossT2) / 2
#         loss = loss1 + loss2 + loss3 + loss4
#         optimizer.zero_grad()
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.2)
#         optimizer.step()

#         loss_value = loss.item()
#         scores = cos_a1 + cos_a2 + cos_b1 + cos_b2
#         # score2 = cos_a2_b2 + cos_b2_a2 + cos_a4_b4 + cos_b4_a4
#         # scores = (score1 + score2) / 2

#         _, max_index = torch.max(scores, 1)
#         acc = 100 * torch.sum(torch.eq(max_index, target)).item() / query_shot / way

#         avg_acc += acc
#         avg_loss += loss_value

#     avg_acc = avg_acc / (i + 1)
#     avg_loss = avg_loss / (i + 1)

#     writer.add_scalar('loss', avg_loss, iter_counter)
#     writer.add_scalar('train_acc', avg_acc, iter_counter)

#     return iter_counter, avg_acc

import torch

from loss import Angular_Isotonic_Loss


def default_train(train_loader, model, optimizer, writer, iter_counter, args):

    if args.gpu_num > 1:
        way = model.module.way
        query_shot = model.module.shots[-1]
        support_shot = model.module.shots[0]

    else:
        way = model.way
        query_shot = model.shots[-1]
    target = torch.LongTensor([i // query_shot for i in range(query_shot * way)]).cuda()

    criterion = Angular_Isotonic_Loss(args.train_way, args.lamda, args.mrg, args.threshold).cuda()

    lr = optimizer.param_groups[0]['lr']
    writer.add_scalar('lr', lr, iter_counter)

    avg_loss = 0
    avg_acc = 0

    for i, (inp, _) in enumerate(train_loader):
        iter_counter += 1

        if args.gpu_num > 1:
            inp_spt = inp[:way * support_shot]
            inp_qry = inp[way * support_shot:]
            qry_num = inp_qry.shape[0]
            inp_list = []
            for i in range(args.gpu_num):
                inp_qry_fraction = inp_qry[int(qry_num/i):int(qry_num/(i+1))]
                inp_list.append(torch.cat((inp_spt, inp_qry_fraction), dim=0))
            inp = torch.cat(inp_list, dim=0)

        inp = inp.cuda()

        cos_a1, cos_a2, cos_b1, cos_b2 = model(inp)
        loss1 = criterion(cos_a1, target)
        loss2 = criterion(cos_a2, target)
        loss3 = criterion(cos_b1, target)
        loss4 = criterion(cos_b2, target)  
        loss = loss1 + loss2 + loss3 + loss4
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.2)
        optimizer.step()

        loss_value = loss.item()
        scores = cos_a1 + cos_a2 + cos_b1 + cos_b2

        _, max_index = torch.max(scores, 1)
        acc = 100 * torch.sum(torch.eq(max_index, target)).item() / query_shot / way

        avg_acc += acc
        avg_loss += loss_value

    avg_acc = avg_acc / (i + 1)
    avg_loss = avg_loss / (i + 1)

    writer.add_scalar('loss', avg_loss, iter_counter)
    writer.add_scalar('train_acc', avg_acc, iter_counter)

    return iter_counter, avg_acc



