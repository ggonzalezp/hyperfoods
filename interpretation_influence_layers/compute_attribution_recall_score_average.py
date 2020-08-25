###Attribution recall score -- alternative computation in which I only generate 1 vector (mean) for all of the samples and then do the GSEA report and find the ratios


from utils_interpretation import *
from glob import glob
import pandas as pd
import numpy as np
import subprocess



#################   Cheb Models  ################################################

# For all layers
# for all splits


for norm in [True, False]:
    for name in ['attributions']:
        for num_layers in [3]:
            gene_names_path = './../dataset/onecc_noiso_string_gene_names.hkl'


            attributions = []
            for split in range(5):
                if norm:
                    outdir = './out_attribution_recall_score/cheb/layers_{}/norm/split_{}/*'.format(num_layers, split) ##dir with attribution vectors for all samples
                else:
                    outdir = './out_attribution_recall_score/cheb/layers_{}/raw/split_{}/*'.format(num_layers, split) ##dir with attribution vectors for all samples
                paths = sorted(glob(outdir))
                for path in paths:
                    l = pd.read_csv(path+'/list.rnk', sep='\t')
                    attributions.append(l['WEIGHT'].values)


            attributions = np.array(attributions)
            attributions = np.mean(attributions, 0)

            gene_ranked_dataframe = pd.concat([pd.DataFrame(l['LABEL']), pd.DataFrame(attributions)], 1)
            gene_ranked_dataframe.columns = ['LABEL', 'WEIGHT']                       #Name columns

            outdir_general = path+'/../../'
            gene_ranked_dataframe.to_csv(outdir_general + 'average_list.rnk', sep = '\t', index = False)
            name = 'average_attributions'
            command = 'bash ./GSEA_Linux_4.0.3/gsea-cli.sh GSEAPreranked -rnk {}/average_list.rnk -gmx ./GSEA_Linux_4.0.3/c2.cp.kegg.v7.0.symbols.gmt -nperm 1000  -out {} -rpt_label {}'.format(outdir_general, outdir_general, name)
            subprocess.call(command, shell = True)




                #Gets ratio OR pathways for all and average
            r = get_ratio_OR_pathways(outdir_general, name)


            log_handle = open(outdir_general+'/{}_recall_score.txt'.format(name), 'w')
            log_handle.write('Positive ORAC vs OR\tPositive ORAC vs AC\tNegative ORAC vs OR\tNegative ORAC vs AC\n')
            log_handle.write('{}\t{}\t{}\t{}'.format(r['pos_orac_or'],
                                                    r['pos_orac_ac'],
                                                    r['neg_orac_or'],
                                                    r['neg_orac_ac']))
            log_handle.close()




#################   GCN Models  ################################################

# For all layers
# for all splits


for norm in [False, True]:
    for name in ['attributions']:
        for num_layers in [1,2,3]:
            gene_names_path = './../dataset/onecc_noiso_string_gene_names.hkl'


            attributions = []
            for split in range(5):
                if norm:
                    outdir = './out_attribution_recall_score/gcn/layers_{}/norm/split_{}/*'.format(num_layers, split) ##dir with attribution vectors for all samples
                else:
                    outdir = './out_attribution_recall_score/gcn/layers_{}/raw/split_{}/*'.format(num_layers, split) ##dir with attribution vectors for all samples

                paths = sorted(glob(outdir))
                for path in paths:
                    l = pd.read_csv(path+'/list.rnk', sep='\t')
                    attributions.append(l['WEIGHT'].values)


            attributions = np.array(attributions)
            attributions = np.mean(attributions, 0)

            gene_ranked_dataframe = pd.concat([pd.DataFrame(l['LABEL']), pd.DataFrame(attributions)], 1)
            gene_ranked_dataframe.columns = ['LABEL', 'WEIGHT']                       #Name columns

            outdir_general = path+'/../../'
            gene_ranked_dataframe.to_csv(outdir_general + 'average_list.rnk', sep = '\t', index = False)
            name = 'average_attributions'
            command = 'bash ./GSEA_Linux_4.0.3/gsea-cli.sh GSEAPreranked -rnk {}/average_list.rnk -gmx ./GSEA_Linux_4.0.3/c2.cp.kegg.v7.0.symbols.gmt -nperm 1000  -out {} -rpt_label {}'.format(outdir_general, outdir_general, name)
            subprocess.call(command, shell = True)




                #Gets ratio OR pathways for all and average
            r = get_ratio_OR_pathways(outdir_general, name)


            log_handle = open(outdir_general+'/{}_recall_score.txt'.format(name), 'w')
            log_handle.write('Positive ORAC vs OR\tPositive ORAC vs AC\tNegative ORAC vs OR\tNegative ORAC vs AC\n')
            log_handle.write('{}\t{}\t{}\t{}'.format(r['pos_orac_or'],
                                                    r['pos_orac_ac'],
                                                    r['neg_orac_or'],
                                                    r['neg_orac_ac']))
            log_handle.close()
