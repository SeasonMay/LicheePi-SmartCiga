# import os
#
#
# def rename1(path):
#     i = 0
#     '该文件夹下所有的文件（包括文件夹）'
#     FileList = os.listdir(path)
#     '遍历所有文件'
#     for files in FileList:
#         '原来的文件路径'
#         oldDirPath = os.path.join(path, files)
#
#         '文件名'
#         fileName = os.path.splitext(files)[0]
#         '文件扩展名'
#         fileType = os.path.splitext(files)[1]
#         '新的文件路径'
#         newDirPath = os.path.join(path, str(i) + fileType)
#         '重命名'
#         os.rename(oldDirPath, newDirPath)
#         i += 1
#
#
# path = r'C:\Users\hp\Desktop\data'
# rename1(path)
import os


class BatchRename():

    def __init__(self):
        self.path = r'C:\Users\hp\Desktop\testv4'  # 图片的路径

    def rename(self):
        filelist = os.listdir(self.path)
        filelist.sort()
        total_num = len(filelist)  # 获取文件中有多少图片
        i = 0  # 文件命名从哪里开始（即命名从哪里开始）
        for item in filelist:
            if item.endswith('.jpg') or item.endswith('.jpeg'):
                src = os.path.join(self.path, item)
                dst = os.path.join(os.path.abspath(self.path),  str(186+i) + '.jpg')

                try:
                    os.rename(src, dst)
                    print('converting %s to %s ...' % (src, dst))
                    i = i + 1
                except Exception as e:
                    print(e)
                    print('rename dir fail\r\n')

        print('total %d to rename & converted %d jpgs' % (total_num, i))


if __name__ == '__main__':
    demo = BatchRename()  # 创建对象
    demo.rename()  # 调用对象的方法
