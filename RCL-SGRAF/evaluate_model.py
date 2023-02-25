import evaluation
import opts
opt = opts.parse_opt()
file_name = opt.model_name + '/' + ('%s_model_best_%g_%g.pth.tar' % (opt.data_name, opt.noise_rate, opt.margin))
# file_name = opt.model_name + '/' + ('%s_model_best_%g_%g_%g.pth.tar' % (opt.data_name, opt.noise_rate, opt.margin, opt.tau))
evaluation.evalrank(file_name, data_path="../SCAN/data/data", split="test")
