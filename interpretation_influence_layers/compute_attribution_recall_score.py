
from utils_interpretation import *






# #################   GCN Models  ################################################

#For all layers
#for all splits



##RAW
#
# norm = False
# for name in ['attributions']:
#     for num_layers in [1,2,3]:
#         pos_orac_or = []
#         pos_orac_ac = []
#         neg_orac_or = []
#         neg_orac_ac = []
#
#         gene_names_path = './../dataset/onecc_noiso_string_gene_names.hkl'
#         #Parameters
#         dataset_dir = './../dataset'
#         hidden_gcn = 8
#         hidden_fc = 32
#
#
#         for split in range(5):
#             splits_dir = './../dataset/data_hyperfoods_drugcentral/5foldCVsplits/{}'.format(split)
#             in_channels = 1
#             n_classes = 2
#             feature_names_path = './../dataset/onecc_noiso_string_gene_names.hkl'
#             model_type = 'gcn'
#             outdir = './out_attribution_recall_score/gcn/layers_{}/raw/split_{}'.format(num_layers, split)
#             ##find the best model for the split from the log
#             model_path = './../out/gcnmodel/final_CV/layers_{}/raw'.format(num_layers)
#             log_path = osp.join(model_path, 'average_cv_performance.txt')
#             log = open(log_path, 'r')
#             best_model_info = log.readlines()[split]
#             best_model_info = best_model_info.split('\t')[2].split('/')[-1]
#             # print(best_model_info)
#             best_model_path = './../out/gcnmodel/final_CV/layers_{}/raw/{}/{}'.format(num_layers, split, best_model_info)
#             do_layers = int(best_model_info.split('_')[-3])
#             batchnorm = best_model_info.split('_')[-1] == 'True'
#             k_hops = None
#
#             if name == 'embeddings':
#                 data_frame = get_embeddings_to_interpret_all_positive_in_test(dataset_dir, outdir, best_model_path, norm, do_layers, batchnorm, num_layers, hidden_gcn, hidden_fc, splits_dir, in_channels, n_classes, feature_names_path, model_type, k_hops)
#             elif name =='attributions':
#                 data_frame = get_attributions_to_interpret_all_positive_in_test(dataset_dir, outdir, best_model_path, norm, do_layers, batchnorm, num_layers, hidden_gcn, hidden_fc, splits_dir, in_channels, n_classes, feature_names_path, model_type, k_hops)
#
#
#             get_gsea_reports(data_frame, outdir, name)
#
#             #Gets ratio OR pathways for all and average
#             for i in range(data_frame.shape[0]):
#                 outdir_i = outdir + '/{}'.format(i)
#                 r = get_ratio_OR_pathways(outdir_i, name)
#                 pos_orac_or.append(r['pos_orac_or'])
#                 pos_orac_ac.append(r['pos_orac_ac'])
#                 neg_orac_or.append(r['neg_orac_or'])
#                 neg_orac_ac.append(r['neg_orac_ac'])
#
#
#         log_handle = open(outdir+'/../{}_recall_score.txt'.format(name), 'w')
#         log_handle.write('Positive ORAC vs OR\tstd\tPositive ORAC vs AC\tstd\tNegative ORAC vs OR\tstd\tNegative ORAC vs AC\tstd\n')
#         log_handle.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(np.mean(pos_orac_or),
#                                                                 np.std(pos_orac_or),
#                                                                 np.mean(pos_orac_ac),
#                                                                 np.std(pos_orac_ac),
#                                                                 np.mean(neg_orac_or),
#                                                                 np.std(neg_orac_or),
#                                                                 np.mean(neg_orac_ac),
#                                                                 np.std(neg_orac_ac)))
#         log_handle.close()





# #################   CHEBNET Models  ################################################

#For all layers
#for all splits



##RAW

norm = False
for name in ['attributions']:
    for num_layers in [1,2,3]:
        pos_orac_or = []
        pos_orac_ac = []
        neg_orac_or = []
        neg_orac_ac = []

        gene_names_path = './../dataset/onecc_noiso_string_gene_names.hkl'
        #Parameters
        dataset_dir = './../dataset'
        hidden_gcn = 8
        hidden_fc = 32


        for split in range(5):
            splits_dir = './../dataset/data_hyperfoods_drugcentral/5foldCVsplits/{}'.format(split)
            in_channels = 1
            n_classes = 2
            feature_names_path = './../dataset/onecc_noiso_string_gene_names.hkl'
            model_type = 'cheb'
            outdir = './out_attribution_recall_score/cheb/layers_{}/raw/split_{}'.format(num_layers, split)
            ##find the best model for the split from the log
            model_path = './../out/chebmodel/final_CV/layers_{}/raw'.format(num_layers)
            log_path = osp.join(model_path, 'average_cv_performance.txt')
            log = open(log_path, 'r')
            best_model_info = log.readlines()[split]
            best_model_info = best_model_info.split('\t')[2].split('/')[-1]
            # print(best_model_info)
            best_model_path = model_path + '/{}/{}'.format(split, best_model_info)

            do_layers = int(best_model_info.split('_')[-5])
            k_hops = int(best_model_info.split('_')[-3])
            batchnorm = best_model_info.split('_')[-1] == 'True'

            if name == 'embeddings':
                data_frame = get_embeddings_to_interpret_all_positive_in_test(dataset_dir, outdir, best_model_path, norm, do_layers, batchnorm, num_layers, hidden_gcn, hidden_fc, splits_dir, in_channels, n_classes, feature_names_path, model_type, k_hops)
            elif name =='attributions':
                data_frame = get_attributions_to_interpret_all_positive_in_test(dataset_dir, outdir, best_model_path, norm, do_layers, batchnorm, num_layers, hidden_gcn, hidden_fc, splits_dir, in_channels, n_classes, feature_names_path, model_type, k_hops)


            get_gsea_reports(data_frame, outdir, name)

            #Gets ratio OR pathways for all and average
            for i in range(data_frame.shape[0]):
                outdir_i = outdir + '/{}'.format(i)
                r = get_ratio_OR_pathways(outdir_i, name)
                pos_orac_or.append(r['pos_orac_or'])
                pos_orac_ac.append(r['pos_orac_ac'])
                neg_orac_or.append(r['neg_orac_or'])
                neg_orac_ac.append(r['neg_orac_ac'])


        log_handle = open(outdir+'/../{}_recall_score.txt'.format(name), 'w')
        log_handle.write('Positive ORAC vs OR\tstd\tPositive ORAC vs AC\tstd\tNegative ORAC vs OR\tstd\tNegative ORAC vs AC\tstd\n')
        log_handle.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(np.mean(pos_orac_or),
                                                                np.std(pos_orac_or),
                                                                np.mean(pos_orac_ac),
                                                                np.std(pos_orac_ac),
                                                                np.mean(neg_orac_or),
                                                                np.std(neg_orac_or),
                                                                np.mean(neg_orac_ac),
                                                                np.std(neg_orac_ac)))
        log_handle.close()




#NORM
norm = True
for name in ['attributions']:
    for num_layers in [1,2,3]:
        pos_orac_or = []
        pos_orac_ac = []
        neg_orac_or = []
        neg_orac_ac = []

        gene_names_path = './../dataset/onecc_noiso_string_gene_names.hkl'
        #Parameters
        dataset_dir = './../dataset'
        hidden_gcn = 8
        hidden_fc = 32


        for split in range(5):
            splits_dir = './../dataset/data_hyperfoods_drugcentral/5foldCVsplits/{}'.format(split)
            in_channels = 1
            n_classes = 2
            feature_names_path = './../dataset/onecc_noiso_string_gene_names.hkl'
            model_type = 'cheb'
            outdir = './out_attribution_recall_score/cheb/layers_{}/norm/split_{}'.format(num_layers, split)
            ##find the best model for the split from the log
            model_path = './../out/chebmodel/final_CV/layers_{}/norm'.format(num_layers)
            log_path = osp.join(model_path, 'average_cv_performance.txt')
            log = open(log_path, 'r')
            best_model_info = log.readlines()[split]
            best_model_info = best_model_info.split('\t')[2].split('/')[-1]
            # print(best_model_info)
            best_model_path = model_path + '/{}/{}'.format(split, best_model_info)
            do_layers = int(best_model_info.split('_')[-5])
            k_hops = int(best_model_info.split('_')[-3])
            batchnorm = best_model_info.split('_')[-1] == 'True'

            if name == 'embeddings':
                data_frame = get_embeddings_to_interpret_all_positive_in_test(dataset_dir, outdir, best_model_path, norm, do_layers, batchnorm, num_layers, hidden_gcn, hidden_fc, splits_dir, in_channels, n_classes, feature_names_path, model_type, k_hops)
            elif name =='attributions':
                data_frame = get_attributions_to_interpret_all_positive_in_test(dataset_dir, outdir, best_model_path, norm, do_layers, batchnorm, num_layers, hidden_gcn, hidden_fc, splits_dir, in_channels, n_classes, feature_names_path, model_type, k_hops)


            get_gsea_reports(data_frame, outdir, name)

            #Gets ratio OR pathways for all and average
            for i in range(data_frame.shape[0]):
                outdir_i = outdir + '/{}'.format(i)
                r = get_ratio_OR_pathways(outdir_i, name)
                pos_orac_or.append(r['pos_orac_or'])
                pos_orac_ac.append(r['pos_orac_ac'])
                neg_orac_or.append(r['neg_orac_or'])
                neg_orac_ac.append(r['neg_orac_ac'])


        log_handle = open(outdir+'/../{}_recall_score.txt'.format(name), 'w')
        log_handle.write('Positive ORAC vs OR\tstd\tPositive ORAC vs AC\tstd\tNegative ORAC vs OR\tstd\tNegative ORAC vs AC\tstd\n')
        log_handle.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(np.mean(pos_orac_or),
                                                                np.std(pos_orac_or),
                                                                np.mean(pos_orac_ac),
                                                                np.std(pos_orac_ac),
                                                                np.mean(neg_orac_or),
                                                                np.std(neg_orac_or),
                                                                np.mean(neg_orac_ac),
                                                                np.std(neg_orac_ac)))
        log_handle.close()





##GCN
#NORM
norm = True
name = 'embeddings'
for name in ['attributions']:
    for num_layers in [1,2,3]:
        pos_orac_or = []
        pos_orac_ac = []
        neg_orac_or = []
        neg_orac_ac = []

        gene_names_path = './../dataset/onecc_noiso_string_gene_names.hkl'
        #Parameters
        dataset_dir = './../dataset'
        hidden_gcn = 8
        hidden_fc = 32


        for split in range(5):
            splits_dir = './../dataset/data_hyperfoods_drugcentral/5foldCVsplits/{}'.format(split)
            in_channels = 1
            n_classes = 2
            feature_names_path = './../dataset/onecc_noiso_string_gene_names.hkl'
            model_type = 'gcn'
            outdir = './out_attribution_recall_score/gcn/layers_{}/norm/split_{}'.format(num_layers, split)
            ##find the best model for the split from the log
            model_path = './../out/gcnmodel/final_CV/layers_{}/norm'.format(num_layers)
            log_path = osp.join(model_path, 'average_cv_performance.txt')
            log = open(log_path, 'r')
            best_model_info = log.readlines()[split]
            best_model_info = best_model_info.split('\t')[2].split('/')[-1]
            # print(best_model_info)
            best_model_path = model_path + '/{}/{}'.format(split, best_model_info)
            do_layers = int(best_model_info.split('_')[-3])
            batchnorm = best_model_info.split('_')[-1] == 'True'
            k_hops = None

            if name == 'embeddings':
                data_frame = get_embeddings_to_interpret_all_positive_in_test(dataset_dir, outdir, best_model_path, norm, do_layers, batchnorm, num_layers, hidden_gcn, hidden_fc, splits_dir, in_channels, n_classes, feature_names_path, model_type, k_hops)
            elif name =='attributions':
                data_frame = get_attributions_to_interpret_all_positive_in_test(dataset_dir, outdir, best_model_path, norm, do_layers, batchnorm, num_layers, hidden_gcn, hidden_fc, splits_dir, in_channels, n_classes, feature_names_path, model_type, k_hops)


            get_gsea_reports(data_frame, outdir, name)

            #Gets ratio OR pathways for all and average
            for i in range(data_frame.shape[0]):
                outdir_i = outdir + '/{}'.format(i)
                r = get_ratio_OR_pathways(outdir_i, name)
                pos_orac_or.append(r['pos_orac_or'])
                pos_orac_ac.append(r['pos_orac_ac'])
                neg_orac_or.append(r['neg_orac_or'])
                neg_orac_ac.append(r['neg_orac_ac'])


        log_handle = open(outdir+'/../{}_recall_score.txt'.format(name), 'w')
        log_handle.write('Positive ORAC vs OR\tstd\tPositive ORAC vs AC\tstd\tNegative ORAC vs OR\tstd\tNegative ORAC vs AC\tstd\n')
        log_handle.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(np.mean(pos_orac_or),
                                                                np.std(pos_orac_or),
                                                                np.mean(pos_orac_ac),
                                                                np.std(pos_orac_ac),
                                                                np.mean(neg_orac_or),
                                                                np.std(neg_orac_or),
                                                                np.mean(neg_orac_ac),
                                                                np.std(neg_orac_ac)))
        log_handle.close()






# #################   SAGE Models  ################################################

#For all layers
#for all splits



##RAW

norm = False
name = 'embeddings'
for name in ['attributions']:
    for num_layers in [1,2,3]:
        pos_orac_or = []
        pos_orac_ac = []
        neg_orac_or = []
        neg_orac_ac = []

        gene_names_path = './../dataset/onecc_noiso_string_gene_names.hkl'
        #Parameters
        dataset_dir = './../dataset'
        hidden_gcn = 8
        hidden_fc = 32


        for split in range(5):
            splits_dir = './../dataset/data_hyperfoods_drugcentral/5foldCVsplits/{}'.format(split)
            in_channels = 1
            n_classes = 2
            feature_names_path = './../dataset/onecc_noiso_string_gene_names.hkl'
            model_type = 'sage'
            outdir = './out_attribution_recall_score/sage/layers_{}/raw/split_{}'.format(num_layers, split)
            ##find the best model for the split from the log
            model_path = './../out/sagemodel/final_CV/layers_{}/raw'.format(num_layers)
            log_path = osp.join(model_path, 'average_cv_performance.txt')
            log = open(log_path, 'r')
            best_model_info = log.readlines()[split]
            best_model_info = best_model_info.split('\t')[2].split('/')[-1]
            # print(best_model_info)
            best_model_path = model_path + '/{}/{}'.format(split, best_model_info)
            do_layers = int(best_model_info.split('_')[-3])
            batchnorm = best_model_info.split('_')[-1] == 'True'
            k_hops = None

            if name == 'embeddings':
                data_frame = get_embeddings_to_interpret_all_positive_in_test(dataset_dir, outdir, best_model_path, norm, do_layers, batchnorm, num_layers, hidden_gcn, hidden_fc, splits_dir, in_channels, n_classes, feature_names_path, model_type, k_hops)
            elif name =='attributions':
                data_frame = get_attributions_to_interpret_all_positive_in_test(dataset_dir, outdir, best_model_path, norm, do_layers, batchnorm, num_layers, hidden_gcn, hidden_fc, splits_dir, in_channels, n_classes, feature_names_path, model_type, k_hops)


            get_gsea_reports(data_frame, outdir, name)

            #Gets ratio OR pathways for all and average
            for i in range(data_frame.shape[0]):
                outdir_i = outdir + '/{}'.format(i)
                r = get_ratio_OR_pathways(outdir_i, name)
                pos_orac_or.append(r['pos_orac_or'])
                pos_orac_ac.append(r['pos_orac_ac'])
                neg_orac_or.append(r['neg_orac_or'])
                neg_orac_ac.append(r['neg_orac_ac'])


        log_handle = open(outdir+'/../{}_recall_score.txt'.format(name), 'w')
        log_handle.write('Positive ORAC vs OR\tstd\tPositive ORAC vs AC\tstd\tNegative ORAC vs OR\tstd\tNegative ORAC vs AC\tstd\n')
        log_handle.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(np.mean(pos_orac_or),
                                                                np.std(pos_orac_or),
                                                                np.mean(pos_orac_ac),
                                                                np.std(pos_orac_ac),
                                                                np.mean(neg_orac_or),
                                                                np.std(neg_orac_or),
                                                                np.mean(neg_orac_ac),
                                                                np.std(neg_orac_ac)))
        log_handle.close()




#NORM
norm = True
name = 'embeddings'
for name in ['attributions']:
    for num_layers in [1,2,3]:
        pos_orac_or = []
        pos_orac_ac = []
        neg_orac_or = []
        neg_orac_ac = []

        gene_names_path = './../dataset/onecc_noiso_string_gene_names.hkl'
        #Parameters
        dataset_dir = './../dataset'
        hidden_gcn = 8
        hidden_fc = 32


        for split in range(5):
            splits_dir = './../dataset/data_hyperfoods_drugcentral/5foldCVsplits/{}'.format(split)
            in_channels = 1
            n_classes = 2
            feature_names_path = './../dataset/onecc_noiso_string_gene_names.hkl'
            model_type = 'sage'
            outdir = './out_attribution_recall_score/sage/layers_{}/norm/split_{}'.format(num_layers, split)
            ##find the best model for the split from the log
            model_path = './../out/sagemodel/final_CV/layers_{}/norm'.format(num_layers)
            log_path = osp.join(model_path, 'average_cv_performance.txt')
            log = open(log_path, 'r')
            best_model_info = log.readlines()[split]
            best_model_info = best_model_info.split('\t')[2].split('/')[-1]
            # print(best_model_info)
            best_model_path = model_path + '/{}/{}'.format(split, best_model_info)
            do_layers = int(best_model_info.split('_')[-3])
            batchnorm = best_model_info.split('_')[-1] == 'True'
            k_hops = None

            if name == 'embeddings':
                data_frame = get_embeddings_to_interpret_all_positive_in_test(dataset_dir, outdir, best_model_path, norm, do_layers, batchnorm, num_layers, hidden_gcn, hidden_fc, splits_dir, in_channels, n_classes, feature_names_path, model_type, k_hops)
            elif name =='attributions':
                data_frame = get_attributions_to_interpret_all_positive_in_test(dataset_dir, outdir, best_model_path, norm, do_layers, batchnorm, num_layers, hidden_gcn, hidden_fc, splits_dir, in_channels, n_classes, feature_names_path, model_type, k_hops)


            get_gsea_reports(data_frame, outdir, name)

            #Gets ratio OR pathways for all and average
            for i in range(data_frame.shape[0]):
                outdir_i = outdir + '/{}'.format(i)
                r = get_ratio_OR_pathways(outdir_i, name)
                pos_orac_or.append(r['pos_orac_or'])
                pos_orac_ac.append(r['pos_orac_ac'])
                neg_orac_or.append(r['neg_orac_or'])
                neg_orac_ac.append(r['neg_orac_ac'])


        log_handle = open(outdir+'/../{}_recall_score.txt'.format(name), 'w')
        log_handle.write('Positive ORAC vs OR\tstd\tPositive ORAC vs AC\tstd\tNegative ORAC vs OR\tstd\tNegative ORAC vs AC\tstd\n')
        log_handle.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(np.mean(pos_orac_or),
                                                                np.std(pos_orac_or),
                                                                np.mean(pos_orac_ac),
                                                                np.std(pos_orac_ac),
                                                                np.mean(neg_orac_or),
                                                                np.std(neg_orac_or),
                                                                np.mean(neg_orac_ac),
                                                                np.std(neg_orac_ac)))
        log_handle.close()
