import numpy as np
import time
import sklearn.metrics as sm
import torch
from torch.utils.data import DataLoader
from dataset import CDVAEDataSet
from loss import discriminative_adjacency_matrix
from model import CDVAE

'''
Load dataset
'''

network_name = 'CDVAE'


def AA(m):
    a = np.sum(m, axis=1)
    zuseracclist = []
    for aai in range(m.shape[0]):
        zuseracclist.append(m[aai][aai] / a[aai])
    b = np.average(zuseracclist)
    return b


if __name__ == '__main__':
    repeat_times = 10
    OAacc_ave = 0.0
    AAacc_ave = 0.0
    kappa_ave = 0.0

    for times in range(repeat_times):
        source_traing_set = CDVAEDataSet(source_train_data, source_train_label)
        target_train_set = CDVAEDataSet(target_train_data, target_train_label)
        target_test_set = CDVAEDataSet(target_test_data, target_test_label)

        source_train_num = source_train_label.shape[0]
        target_train_num = target_train_label.shape[0]
        target_test_num = target_test_label.shape[0]
        source_train_loader = DataLoader(dataset=source_traing_set,
                                         batch_size=source_train_num,
                                         shuffle=False)
        target_train_loader = DataLoader(dataset=target_train_set,
                                         batch_size=target_train_num,
                                         shuffle=False)
        target_test_loader = DataLoader(dataset=target_test_set,
                                        batch_size=target_test_num,
                                        shuffle=False)

        source_imput_dim = source_train_data.shape[1]
        target_imput_dim = target_train_data.shape[1]
        hidden_dim = 50
        r = 20
        epoch = 1000
        net = CDVAE(
            source_imput_dim,
            target_imput_dim,
            hidden_dim,
            r,
            class_num)
        for i, (target_data, target_label) in enumerate(target_train_loader):
            source_data, source_label = iter(source_train_loader).next()
            W = discriminative_adjacency_matrix(source_label, target_label)

        optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
        criterion_mse, criterion_crossentropy, criterion_graphloss = net.loss()
        since = time.time()
        with open(data_name + 'test_cla_rate.txt', 'a') as f:
            f.write('\n\n' + network_name + ' r' + str(r) + '  ' + str(
                time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))) + ' The ' + str(
                times) + ' time' + '\n')
        f.close()

        net.train()
        for epoch in range(epoch):
            for i, (target_data, target_label) in enumerate(
                    target_train_loader):
                source_data, source_label = iter(source_train_loader).next()

                optimizer.zero_grad()

                source_hidden_1, source_input_hat_1, source_kld_1, source_hidden_2, source_input_hat_2, source_kld_2, \
                target_hidden_1, target_input_hat_1, target_kld_1, target_hidden_2, target_input_hat_2, target_kld_2, \
                source_out, target_out = net(source_data, target_data)

                source_label = source_label.long()
                target_label = target_label.long()

                alpha1 = (torch.norm(source_data)) ** 2 / \
                         (torch.norm(target_data)) ** 2

                alpha2 = (torch.norm(source_hidden_1)) ** 2 / \
                         (torch.norm(target_hidden_1)) ** 2

                beta = 10

                lambda_ = 0.001

                theta = 1000

                loss_source_vae_firstl = criterion_mse(
                    source_input_hat_1, source_data) + source_kld_1

                loss_target_vae_firstl = criterion_mse(
                    target_input_hat_1, target_data) + target_kld_1

                loss_source_vae_secondl = criterion_mse(
                    source_input_hat_2, source_hidden_1) + source_kld_2

                loss_target_vae_secondl = criterion_mse(
                    target_input_hat_2, target_hidden_1) + target_kld_2

                loss_vae_firstl = loss_source_vae_firstl + loss_target_vae_firstl * alpha1

                loss_vae_secondl = loss_source_vae_secondl + loss_target_vae_secondl * alpha2

                loss_gr_firstl = criterion_graphloss(
                    source_hidden_1, target_hidden_1, W, class_num)

                loss_gr_secondl = criterion_graphloss(
                    source_hidden_2, target_hidden_2, W, class_num)

                loss_fe_firstl = loss_vae_firstl + loss_gr_firstl * lambda_

                loss_fe_secondl = (
                                          loss_vae_secondl + loss_gr_secondl * lambda_) * beta

                loss_c_source = criterion_crossentropy(
                    source_out, source_label)

                loss_c_target = criterion_crossentropy(
                    target_out, target_label)

                loss_c = (loss_c_source + loss_c_target) * theta

                loss = loss_fe_firstl + loss_fe_secondl + loss_c

                loss.backward()
                optimizer.step()

            if (epoch + 1) % 100 == 0:
                print('epoch', epoch + 1, 'loss', loss.item())
                net.eval()
                correct = 0
                total = 0.0
                predlabel = torch.Tensor(np.array([]))
                realtestlabel = torch.Tensor(np.array([]))
                with torch.no_grad():
                    for (target_data, target_label) in target_test_loader:
                        source_data, source_label = iter(source_train_loader).next()
                        source_label = source_label.long()
                        target_label = target_label.long()
                        source_hidden_1, source_input_hat_1, source_kld_1, source_hidden_2, source_input_hat_2, source_kld_2, \
                        target_hidden_1, target_input_hat_1, target_kld_1, target_hidden_2, target_input_hat_2, target_kld_2, \
                        source_out, target_out = net(source_data, target_data)
                        i, predicted = torch.max(target_out.data, dim=1)
                        predlabel = torch.cat((predlabel, predicted), -1)
                        realtestlabel = torch.cat((realtestlabel, target_label), -1)
                        total += target_label.size(0)
                        correct += ((predicted == target_label).sum())
                    test_acc = float(correct) / total
                    predlabel = predlabel + 1
                    realtestlabel = realtestlabel + 1

                    C = sm.confusion_matrix(realtestlabel.data.cpu().numpy(), predlabel.cpu().numpy())
                    kappa = sm.cohen_kappa_score(realtestlabel.data.cpu().numpy(), predlabel.cpu().numpy(), labels=None,
                                                 weights=None, sample_weight=None)
                    aa = AA(C)
                with open(data_name + 'test_cla_rate.txt', 'a') as f:
                    f.write(str(epoch + 1) +
                            '  OA: ' + str(test_acc) +
                            '  AA: ' + str(aa) +
                            '  kappa: ' + str(kappa) +
                            '\n')
                f.close()

        time_last = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_last // 60, time_last % 60))

        OAacc_ave = test_acc + OAacc_ave
        AAacc_ave = aa + AAacc_ave
        kappa_ave = kappa + kappa_ave

    OAacc_ave = (OAacc_ave / repeat_times)
    AAacc_ave = (AAacc_ave / repeat_times)
    kappa_ave = (kappa_ave / repeat_times)
    print(
        str(r) +
        '-dimensional OAaccuracy: ' +
        str(OAacc_ave) +
        ', AAaccuracy: ' +
        str(AAacc_ave) +
        ', kappa: ' +
        str(kappa_ave) +
        ' after ' +
        str(repeat_times) +
        ' averages')
    print('-------------------')
    print('finish!')
