# auto_modeling
auto_modeling
1.程序用途：
	本程序用于满足自动化建模需求。目前版本仅支持表结构不变的情况下的自动二分类模型训练。
	最初的训练需要人工输入命令，之后每月只需更新训练数据，就可以自动训练(定时任务)，和自动预测(predict)。
	也可以手动输入命令训练和预测。
	
2.程序思路：
	最初由于没有模型，所以要人工输入命令训练。输入数据路径及Label名称后，程序训练出模型，
	并将结果(各个列的直方图，箱型图，IV，std，importance等等)输出到result文件下。
	数据清洗过程：先处理缺失数据==>处理异常数据
	特种工程：方差过滤==>Spearman过滤==>卡方检验过滤==>随机森林过滤==>VIF过滤==>IV过滤
	由于数据源是随机的，多样的，所以数据预处理阶段的效果不如人工处理来得好。
	第一次训练完后，将选择的特征和Label和index名称按顺序存到columns和label和index文件中。之后每次自动
	训练模型，程序都从这两个文件读取特征选择信息和Label信息和index信息，自动完成训练。
	流程：第一个月放入训练数据及预测数据==>获得第一个月预测数据==>第二月获取了第一个月的真实数据，工作人员手动更新训练数据(或写脚本自动从数据库导数据)
	==>模型自动训练，预测第二个月的结果==>第三个月获取了第二个月真实数据...==>预测第四个月结果...==>...

3.程序运行方式：
	3.1 第一次训练，或者需要手动训练新模型时候，运行./core/train.py文件，train.py包含以下参数
	######################
	【--label】 type=str  ！第一次训练必须填写，为Label列的列名(也就是Y的名称)(程序自动运行后可以不填)
    【--missing_data_rate】 type=float,default=0.03	！缺失数据阈值，默认0.03，低于阈值的缺失值直接删除，高于的用随机森林算法填补
    【--error_data_rate】,type=float,default=0.01	！异常数据阈值，默认0.01，由于数据的多样性，默认异常数据占比低于0.01时删除，高于0.01可能并非异常数据保留，建议设置更低一些。
    【--randomforest_treshold】 type=float,default=0.01	！随机森林选择特征阈值，默认0.01，高于0.01的重要性保留，低于的删除
    【--iv_treshold】 type=float,default=0.02	！IV特征选择阈值，默认0.02，高于的保留，低于的删除
    【--iv_cut】 type=int,default=5		！IV特征选择中数据转换用的手动等距分箱数目(优先自动等频分箱，失败后才转为手动分箱)，默认5箱
    【--split_rate】 type=float,default=0.25	！训练数据的训练-测试划分比值，默认0.25，不建议设太高
	【--index_col】 type=str,default='Unnamed: 0'	！作为index_col不参与建模的列名，默认‘Unnamed: 0’，如果数据集没有index，就不考虑。【若有index，一定要填对，区分大小写和空格】
	例如，在训练信用卡违约可能的模型中，Label名字为SeriousDlqin2yrs，index列的名称为'Unnamed: 0'第一次训练时候：
	python ./auto_modeling/core/train.py --label SeriousDlqin2yrs --missing_data_rate 0.03 --index_col Unnamed: 0
	######################
	请将训练数据放置于 ../auto_modeling/data/train_data 目录下，并且建议用【.csv】格式储存，其他格式会有乱码等读取问题。
	训练数据中必须包含列名！且将label也一并【合并在一个文件中】，【不要】将训练数据和标签分为两个文件
	../auto_modeling/data/train_data 目录下不要放置多于1个文件数，否则识别报错
	
	3.2 程序自动运行，设置定时任务，在需要的时间自动运行train.py 文件，只需python train.py 就好，程序会自动读取
	../auto_modeling/model目录下的特征选择column及label及index文件和新的训练数据进行训练
	
	3.3 预测数据，需要手动输入命令预测数据，如果需要的话，可以设置定时任务自动预测，运行./core/predict.py文件，predict.py包含以下参数
	######################
	【--model_name】 type=str,default='pickle_model.dat'	！模型名称，默认为'pickle_model.dat'，若要使用别的名称直接输入，不建议这样。
	例如，在预测信用卡违约可能的模型中，模型名用默认的
	python ./auto_modeling/core/predict.py
	######################
	请将预测数据放置于 ../auto_modeling/data/test_data 目录下，并且建议用【.csv】格式储存，其他格式会有乱码等读取问题。
	训练数据中必须包含特征列名！
	../auto_modeling/data/test_data 目录下不要放置多于1个文件数，否则识别报错
	
4. 程序输出：
	输出文件包括：
	4.1 log目录下的运行日志文件：
			train_log.log
			predict_log.log
	4.2 model目录下的文件：
			columns  			#特征选择后的列名【不要删除这个文件！】
			label				#需要预测的列名(Y)【不要删除这个文件！】
			index				#需要执行的index列名【不要删除这个文件！】
			woe_cut				#predict时候将原始数据分箱的依据【不要删除这个文件！】
			pickle_model.dat	#train后的模型【不要删除这个文件！】
	4.3 result目录下的文件：
			*.png				#各列训练数据的直方图和箱线图，方便人工筛选数据
			record.txt			#各列数据的各种统计指标，例如std、IV、WOE等等
			predict_result.csv	#predict的结果文件
	
5. 文件目录简介：
	core：源代码的文件
	data: {		train_data :放置训练用数据
				test_data  :放置预测用数据
			}
	log：日志配置文件和日志
	model：放置模型及Label和column和index结果
	result：放置各种指标及预测结果
	
6. 版本：
	version 0.1		#初步完成代码
	version 0.2		#修复部分bug，如woe出现inf和nan情况,concat是索引问题
					#更改一些逻辑错误，如数据清洗顺序等
					#调参优化，把随机森林阈值调至0.01，错误率阈值0.01
					#增加index识别，让训练前正确去除index列
	version 0.3		#修复predict过拟合bug
					#增加woe_cut文件，使predict更稳定
					
	
