# desc: str = 'MyMemory Language Translator is an application that lets users to translate text from one language to another. This application uses the MyMemory API service.'
# sentences = [sen.strip().lower() for sen in desc.strip().split('.') if sen != '']
# print(sentences)
import pandas
# api_df = pd.read_csv('./cloud api dataset/apis_detail.csv', encoding='ISO-8859-1').loc[:, ['url', 'description']]
# print(api_df.head())

######################################################################
# 统计一下有多少个api被调用了
# api_set = set()
# with open(file='./origin dataset/inter.csv', mode='r') as fp:
#     for line in fp.readlines():
#         l = line.strip('\n').split(',')
#         mashup_id = l[0]
#         api_list = [int(api) for api in l[1:]]
#         for api in api_list:
#             if api not in api_set:
#                 api_set.add(api)
#
# # 1210 个
# print('the num of api be invoked by mashup: %d' % len(api_set))
######################################################################

# with open(file='./origin dataset/inter.csv', mode='r') as fp:
#     for l in fp.readlines():
#         l = l.strip('\n').split(',')
#         ma_id = l[0]
#         app_id = [api for api in l[1:]]
#         print(type(ma_id))
#         print(type(app_id[0]))
#         break

######################################################################
# 测试config.tpl_list_length是什么意思？
# max_length: int = -1
# file_list = os.listdir('./app train data/')
# for file in file_list:
#     with open(file = './app train data/' + file, mode='r') as fp:
#         for l in fp.readlines():
#             obj = json.loads(l.strip('\n'))
#             max_length = max(len(obj['tpl_list']), max_length)
#
# print(f'max length is : {max_length}')      # 57

######################################################################

# tpl_num_list = [55, 325, 1252, 1622, 2833, 281, 96, 1064, 9, 1448, 2827, 2838, 120, 2717, 237, 424, 28, 12, 90, 217, 76, 12, 63, 973, 17, 99, 61, 34, 38, 84, 29, 158, 14, 257, 37, 19, 304, 2020, 847, 999, 1072, 2645, 3469, 219, 221, 28, 258, 533, 2105, 412, 694, 679, 130, 1690, 10, 161, 125, 16, 125, 26, 142, 64, 460, 261, 54, 469, 153, 22, 20, 35, 19, 15, 176, 103, 100, 20, 104, 252, 370, 337, 3013, 134, 67, 48220, 21998, 948, 262, 157, 23, 3656, 196, 254, 389, 219, 171, 2375, 14, 46, 1279, 196, 9, 14, 6, 26, 14, 99, 30, 34, 10175, 4209, 617, 6848, 1055, 242, 33, 227, 251, 684, 15, 33, 4939, 407, 115, 596, 1934, 147, 110, 22, 7, 696, 27, 9, 139, 72, 15, 291, 55, 115, 21, 839, 462, 169, 52, 14, 407, 115, 27, 34, 23, 70, 12, 791, 95, 14833, 329, 153, 385, 112, 266, 112, 8, 2473, 44, 14, 57, 13, 423, 1821, 19, 111, 10551, 20, 569, 581, 1090, 363, 15, 20, 205, 245, 42, 37, 9, 2469, 111, 6, 42, 22, 40, 160, 169, 151, 251, 744, 20, 78, 436, 63, 385, 711, 81, 184, 154, 5775, 1460, 664, 2104, 202, 2000, 20, 150, 30, 11, 35, 27, 151, 7, 13, 23, 68, 24, 354, 101, 158, 93, 32, 20, 11, 71, 279, 56, 144, 18, 18, 159, 49, 138, 166, 401, 8, 1950, 329, 16, 130, 210, 354, 27, 37, 70, 113, 3352, 2190, 23, 3819, 23831, 84, 2531, 151, 7785, 71, 63, 29, 39, 40, 164, 8762, 13, 79, 17254, 38, 132, 69, 37, 16, 152, 667, 407, 512, 111, 288, 65, 3077, 121, 15, 166, 166, 56, 166, 144, 296, 42578, 6614, 1389, 1389, 165, 853, 2137, 30, 4951, 1976, 13026, 593, 779, 51331, 271, 22838, 55, 4959, 230, 202, 983, 924, 506, 102, 392, 222, 11, 26, 863, 78, 4379, 226, 87, 15, 10, 94, 113, 460, 100, 37, 300, 37, 637, 121, 287, 45, 101, 169, 996, 25, 727, 4154, 63, 882, 26, 17, 28, 3639, 104, 48, 658, 50, 117, 80, 893, 116, 1169, 693, 540, 47, 271, 25, 1670, 40, 121, 66, 718, 28, 370, 3147, 97, 393, 98, 765, 23, 55, 151, 12, 135, 35, 67, 54, 929, 1551, 214, 25, 127, 1819, 590, 29, 507, 16, 28, 3207, 75, 6, 23, 884, 71, 2600, 13, 49, 188, 16, 106, 203, 148, 80, 71, 2100, 249, 86, 279, 89, 8, 2151, 2230, 106, 46, 829, 70, 1665, 140, 24, 459, 730, 111, 12, 40, 53, 315, 24, 10, 650, 255, 25, 29, 57, 52, 133, 7, 63, 24, 180, 31, 194, 167, 141, 26, 11, 114, 159, 205, 29, 124, 220, 16, 398, 177, 42, 6, 924, 47, 33, 12, 352, 20, 31, 71, 185, 249, 299, 76, 48, 1904, 1593, 52, 626, 879, 22, 2225, 282, 175, 659, 695, 337, 13, 794, 897, 4037, 499, 229, 48, 28, 33, 419, 57, 169, 73, 17, 30, 43, 25, 21, 52, 1175, 1051, 46, 6815, 1197, 32, 13, 12, 844, 3807, 7, 94, 209, 477, 12, 152, 211, 4542, 3904, 1030, 170, 17, 58, 1278, 655, 63, 56, 195, 316, 1279, 32, 18, 31, 770, 12, 13, 1727, 168, 1487, 1628, 595, 1103, 28, 1213, 119, 21, 85, 9, 81, 14, 33, 13, 14, 1859, 272, 118, 64, 541, 721, 74, 8, 35, 100, 152, 16, 550, 42, 6, 30, 4616, 1169, 122, 36, 17, 325, 46, 9, 48, 66, 729, 733, 923, 238, 35, 24, 14, 101, 48, 225, 1701, 16, 18, 8, 86, 961, 273, 898, 18, 67, 269, 161, 3501, 158, 407, 268, 248, 256, 89, 30, 102, 65, 24, 91, 1222, 36, 2181, 255, 364, 255, 16, 876, 23, 475, 133, 4599, 35, 42, 40, 30, 5614, 1621, 4838, 21, 41, 54, 154, 2874, 27, 426, 108, 114, 221, 6, 162, 9, 20, 365, 19, 392, 57, 1457, 1219, 88, 146, 77, 88, 117, 80, 67, 104, 71, 2066, 99, 6, 28, 33, 46, 14, 1132, 153, 39, 59, 79, 49, 61, 16, 66, 98, 308, 13, 203, 33, 87, 25, 55, 331, 11, 1142, 5025, 11, 8, 9, 14, 14, 35, 944, 955, 87, 15, 15, 104, 199, 109, 38, 18669, 10245, 470, 1245, 843, 19, 679, 119, 51, 53, 15, 17, 209, 1274, 21, 44, 132, 260, 20, 2742, 24, 945, 2917, 32, 10, 133, 35, 107, 41, 47, 297, 70, 9, 13, 65, 10, 36, 2460, 75, 57, 40, 14, 196, 835, 143, 13, 16, 2396, 890, 24, 262, 104, 410, 7175]
# file_list = os.listdir('./app train data/')
# for file in file_list:
#     with open(file = './app train data/' + file, mode='r') as fp:
#         for l in fp.readlines():
#             obj = json.loads(l.strip('\n'))
#             for tpl in obj['tpl_list']:
#                 if tpl not in tpl_num_list:
#                     print('出现了！结束')
#                     exit()
# print('没有出现，结束')

######################################################################
# 统计一下有多少个tpl被使用
# tpl_num_list = [55, 325, 1252, 1622, 2833, 281, 96, 1064, 9, 1448, 2827, 2838, 120, 2717, 237, 424, 28, 12, 90, 217, 76, 12, 63, 973, 17, 99, 61, 34, 38, 84, 29, 158, 14, 257, 37, 19, 304, 2020, 847, 999, 1072, 2645, 3469, 219, 221, 28, 258, 533, 2105, 412, 694, 679, 130, 1690, 10, 161, 125, 16, 125, 26, 142, 64, 460, 261, 54, 469, 153, 22, 20, 35, 19, 15, 176, 103, 100, 20, 104, 252, 370, 337, 3013, 134, 67, 48220, 21998, 948, 262, 157, 23, 3656, 196, 254, 389, 219, 171, 2375, 14, 46, 1279, 196, 9, 14, 6, 26, 14, 99, 30, 34, 10175, 4209, 617, 6848, 1055, 242, 33, 227, 251, 684, 15, 33, 4939, 407, 115, 596, 1934, 147, 110, 22, 7, 696, 27, 9, 139, 72, 15, 291, 55, 115, 21, 839, 462, 169, 52, 14, 407, 115, 27, 34, 23, 70, 12, 791, 95, 14833, 329, 153, 385, 112, 266, 112, 8, 2473, 44, 14, 57, 13, 423, 1821, 19, 111, 10551, 20, 569, 581, 1090, 363, 15, 20, 205, 245, 42, 37, 9, 2469, 111, 6, 42, 22, 40, 160, 169, 151, 251, 744, 20, 78, 436, 63, 385, 711, 81, 184, 154, 5775, 1460, 664, 2104, 202, 2000, 20, 150, 30, 11, 35, 27, 151, 7, 13, 23, 68, 24, 354, 101, 158, 93, 32, 20, 11, 71, 279, 56, 144, 18, 18, 159, 49, 138, 166, 401, 8, 1950, 329, 16, 130, 210, 354, 27, 37, 70, 113, 3352, 2190, 23, 3819, 23831, 84, 2531, 151, 7785, 71, 63, 29, 39, 40, 164, 8762, 13, 79, 17254, 38, 132, 69, 37, 16, 152, 667, 407, 512, 111, 288, 65, 3077, 121, 15, 166, 166, 56, 166, 144, 296, 42578, 6614, 1389, 1389, 165, 853, 2137, 30, 4951, 1976, 13026, 593, 779, 51331, 271, 22838, 55, 4959, 230, 202, 983, 924, 506, 102, 392, 222, 11, 26, 863, 78, 4379, 226, 87, 15, 10, 94, 113, 460, 100, 37, 300, 37, 637, 121, 287, 45, 101, 169, 996, 25, 727, 4154, 63, 882, 26, 17, 28, 3639, 104, 48, 658, 50, 117, 80, 893, 116, 1169, 693, 540, 47, 271, 25, 1670, 40, 121, 66, 718, 28, 370, 3147, 97, 393, 98, 765, 23, 55, 151, 12, 135, 35, 67, 54, 929, 1551, 214, 25, 127, 1819, 590, 29, 507, 16, 28, 3207, 75, 6, 23, 884, 71, 2600, 13, 49, 188, 16, 106, 203, 148, 80, 71, 2100, 249, 86, 279, 89, 8, 2151, 2230, 106, 46, 829, 70, 1665, 140, 24, 459, 730, 111, 12, 40, 53, 315, 24, 10, 650, 255, 25, 29, 57, 52, 133, 7, 63, 24, 180, 31, 194, 167, 141, 26, 11, 114, 159, 205, 29, 124, 220, 16, 398, 177, 42, 6, 924, 47, 33, 12, 352, 20, 31, 71, 185, 249, 299, 76, 48, 1904, 1593, 52, 626, 879, 22, 2225, 282, 175, 659, 695, 337, 13, 794, 897, 4037, 499, 229, 48, 28, 33, 419, 57, 169, 73, 17, 30, 43, 25, 21, 52, 1175, 1051, 46, 6815, 1197, 32, 13, 12, 844, 3807, 7, 94, 209, 477, 12, 152, 211, 4542, 3904, 1030, 170, 17, 58, 1278, 655, 63, 56, 195, 316, 1279, 32, 18, 31, 770, 12, 13, 1727, 168, 1487, 1628, 595, 1103, 28, 1213, 119, 21, 85, 9, 81, 14, 33, 13, 14, 1859, 272, 118, 64, 541, 721, 74, 8, 35, 100, 152, 16, 550, 42, 6, 30, 4616, 1169, 122, 36, 17, 325, 46, 9, 48, 66, 729, 733, 923, 238, 35, 24, 14, 101, 48, 225, 1701, 16, 18, 8, 86, 961, 273, 898, 18, 67, 269, 161, 3501, 158, 407, 268, 248, 256, 89, 30, 102, 65, 24, 91, 1222, 36, 2181, 255, 364, 255, 16, 876, 23, 475, 133, 4599, 35, 42, 40, 30, 5614, 1621, 4838, 21, 41, 54, 154, 2874, 27, 426, 108, 114, 221, 6, 162, 9, 20, 365, 19, 392, 57, 1457, 1219, 88, 146, 77, 88, 117, 80, 67, 104, 71, 2066, 99, 6, 28, 33, 46, 14, 1132, 153, 39, 59, 79, 49, 61, 16, 66, 98, 308, 13, 203, 33, 87, 25, 55, 331, 11, 1142, 5025, 11, 8, 9, 14, 14, 35, 944, 955, 87, 15, 15, 104, 199, 109, 38, 18669, 10245, 470, 1245, 843, 19, 679, 119, 51, 53, 15, 17, 209, 1274, 21, 44, 132, 260, 20, 2742, 24, 945, 2917, 32, 10, 133, 35, 107, 41, 47, 297, 70, 9, 13, 65, 10, 36, 2460, 75, 57, 40, 14, 196, 835, 143, 13, 16, 2396, 890, 24, 262, 104, 410, 7175]
# tpl_set = set()
# cnt_0: int = 0
# for tpl in tpl_num_list:
#     if tpl == 0:
#         cnt_0 += 1
#
# print(cnt_0)
######################################################################
# 重写cloud api编号, 1-1210的api

# def read_data(file: str = './origin dataset/cloud api detail.csv') -> Dict[int, str]:
#     ret_data: Dict[int, str] = dict()
#     api_df = pd.read_csv(file, encoding='utf8').loc[:, ['id', 'url']]
#     for idx, row in api_df.iterrows():
#         ret_data[int(row['id'])] = row['url']
#     return ret_data
#
#
# # 因为api数据只有1210个，所以需要对inter中和detail中重新赋值一下
# new_api_list: List[int] = []
# old_id2new_id: Dict[int, int] = {}
# with open(file='./origin dataset/inter.csv', mode='r') as fp:
#     for inter in fp.readlines():
#         inter = inter.strip('\n').split(',')
#         mashup_id = int(inter[0])
#         api_list = [int(api) for api in inter[1:]]
#         for api in api_list:
#             if api not in new_api_list:
#                 new_api_list.append(api)
#
# new_api_list = sorted(new_api_list, reverse=False)
#
# for i in range(len(new_api_list)):
#     old_id2new_id[new_api_list[i]] = i + 1

# old_id2_url = read_data()
#
# with open(file='./origin dataset/cloud api detail.csv', mode='w') as fp:
#     fp.write('id,url')
#     for i in range(len(new_api_list)):
#         old_id = new_api_list[i]
#         url = old_id2_url[old_id]
#         new_pair = str(old_id2new_id[old_id]) + ',' + url + '\n'
#         fp.write(new_pair)
# inter.csv文件也需要重写
# w_fp = open(file='./origin dataset/inter.csv', mode='w')
# with open(file='./origin dataset/inter.csv', mode='r') as fp:
#     for pair in fp.readlines():
#         pair = pair.strip('\n').split(',')
#         mashup_id = int(pair[0])
#         api_list = [int(api) for api in pair[1:]]
#         for i in range(len(api_list)):
#             api_list[i] = old_id2new_id[api_list[i]]
#         pair_list = [str(mashup_id)]
#         pair_list.extend([str(api) for api in api_list])
#         content = ','.join(pair_list) + '\n'
#         w_fp.write(content)
#
# new_fp = open(file='./origin dataset/relation.json', mode='w')
# with open(file='./origin dataset/relation.json', mode='r') as fp:
#     for l in fp.readlines():
#         l = l.strip('\n')
#         obj = json.loads(l)
#         api_list = obj['api_list']
#         for i in range(len(api_list)):
#             api_list[i] = old_id2new_id[api_list[i]]
#         obj['api_list'] = api_list
#         obj_s = json.dumps(obj) + '\n'
#         new_fp.write(obj_s)
####################################################################################
# 统计一下api为1 的个数
# cnt_1: int = 0
# with open(file='./origin dataset/relation.json', mode='r') as fp:
#     for l in fp.readlines():
#         obj = json.loads(l.strip('\n'))
#         api_list = obj['api_list']
#         for api in api_list:
#             if api == 1:
#                 cnt_1 += 1
# print(cnt_1)
####################################################################################
# 测试所有的api是否被调用了
# used_dict = dict()
# sum_num: int = 0
# with open(file='./origin dataset/relation.json', mode='r') as fp:
#     for line in fp.readlines():
#         line = line.strip('\n')
#         obj = json.loads(line)
#         for api in obj['api_list']:
#             if api not in used_dict.keys():
#                 used_dict[api] = 1
#                 sum_num += 1
# print(sum_num)
####################################################################################
# tpl_range 是最大的被使用tpl编号, 测试是不是
# max_tpl: int = 0
# file_list = os.listdir('./training dataset/')
# for file in file_list:
#     with open(file='./training dataset/' + file, mode='r') as fp:
#         for line in fp.readlines():
#             obj = json.loads(line.strip('\n'))
#             for tpl in obj['tpl_list']:
#                 max_tpl = max(max_tpl, tpl)
#
# print(max_tpl)
####################################################################################
# 来生成1-1210编号api的被调用个数列表
# api_num_list = [0]*1210
# with open(file='./origin dataset/relation.json', mode='r') as fp:
#     for line in fp.readlines():
#         obj = json.loads(line.strip('\n'))
#         for api in obj['api_list']:
#             api_num_list[api-1] += 1
#
# print(api_num_list)
####################################################################################
# data = {
#     'a': [[1, 2, 3, 4]],
#     'b': [[2, 3, 4, 5]]
# }
# a_tensor = torch.tensor(data['a'], dtype=torch.float64).cuda()
# b_tensor = torch.tensor(data['b'], dtype=torch.float64).cuda()
#
# print('a \'s shape: ', a_tensor.shape)
# print('b \'s shape: ', b_tensor.shape)
# c = torch.cat((a_tensor, b_tensor), dim=1)
#
# print(c.shape)
# print(c)
####################################################################################
# 测试一下DataLoader返回的是什么, 是否会自动合并为tensor
# class MyDataSet(Dataset):
#     def __init__(self):
#         super(MyDataSet, self).__init__()
#         file_path = './training data/train_0_torch.json'
#         fp = open(file=file_path, mode='r')
#         lines = fp.readlines()
#         self.size = len(lines)
#         self.file = lines
#         fp.close()
#
#     def __len__(self):
#         return self.size
#
#     def __getitem__(self, item_idx):
#         test_obj = json.loads(self.file[item_idx].strip('\n'))
#
#         return test_obj
#
#
# dataset = MyDataSet()
# loader = DataLoader(dataset=dataset, shuffle=True, batch_size=64)
#
# for idx, data in enumerate(loader):
#     print(idx)
#     print(data)
#     break
####################################################################################
# import pandas as pd
# from typing import List
# # 测试cloud api dataset 下的mashups_detail是不是有空的情况
# mashup_pd = pd.read_csv('./cloud api dataset/mashups_detail.csv', encoding='ISO-8859-1')
# desc_na_cnt: int = 0
# cate_na_cnt: int = 0
# category_list: List[str] = []
# for index, row in mashup_pd.iterrows():
#     if pd.isna(row['description']):
#         desc_na_cnt += 1
#     if pd.isna(row['categories']):
#         cate_na_cnt += 1
#         category_list += [row['title']]
#
# print(f'description 为空的个数{desc_na_cnt}')
# print(f'categories 为空的个数{cate_na_cnt}')
# print(category_list)
####################################################################################
# import pandas as pd
# from typing import List, Dict, Tuple
#
#
# def get_api_description_and_tag() -> Tuple[Dict[str, str], Dict[str, str]]:
#     api_df = pd.read_csv('./cloud api dataset/apis_detail.csv', encoding='ISO-8859-1')
#     api_desc_dict: Dict[str, str] = {}
#     api_tag_dict: Dict[str, str] = {}
#     for index, row in api_df.iterrows():
#         row['url'] = row['url'].strip('\n')
#         api_desc_dict[row['url']] = row['description']
#         api_tag_dict[row['url']] = row['tags']
#
#     del api_df
#     return api_desc_dict, api_tag_dict
#
#
# # 需要重新新建一个文件
# if __name__ == '__main__':
#     desc_dict, tag_dict = get_api_description_and_tag()
#     new_api_details_data: List = []
#     api_pd = pd.read_csv('./origin dataset/cloud api detail.csv')
#     for index, row in api_pd.iterrows():
#         api_url = row['url']
#         new_api_details_data.append({
#             'id': index+1,
#             'url': api_url,
#             'category': tag_dict[api_url],
#             'description': desc_dict[api_url]
#         })
#
#     # 创建DataFrame
#     new_data_frame = pd.DataFrame(new_api_details_data, columns=['id', 'url', 'category', 'description'])
#     new_data_frame.to_csv(path_or_buf='./new_api_details.csv', index=False)
####################################################################################
# 测试new_api_details中category和description是否有为空
# import pandas as pd
# from typing import List
#
# # api details需要ISO-8859-1编码方式打开
# # api details 的问题已经解决了
# new_api_df = pd.read_csv('./new_api_details.csv', encoding='ISO-8859-1')
# cate_na_cnt: int = 0
# desc_na_cnt: int = 0
# api_list: List[str] = []
# for index, row in new_api_df.iterrows():
#     if pd.isna(row['category']):
#         cate_na_cnt += 1
#         api_list += [row['id']]
#     if pd.isna(row['description']):
#         desc_na_cnt += 1
# print(f'category为nan的个数: {cate_na_cnt}')
# print(f'description为nan的个数: {desc_na_cnt}')
# print(api_list)
####################################################################################
# line_cnt: int = 0
# with open(file='./new origin dataset/mashups_details.csv', encoding='ISO-8859-1') as f:
#     for line in f.readlines():
#         print(line)
#         line_cnt += 1
#         if line_cnt == 10:
#             break
