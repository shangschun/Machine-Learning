# -*- coding: utf-8 -*- 2
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib as mpl
from pprint import pprint
# sigma:奇异值
# u:左特征向量
# v:右特征向量
# K:取多少特征
def restore1(sigma,u,v,K):
    # 获得左特征向量的长度
    m = len(u)
    # 获取右特征的额长度
    n = len(v[0])
    a = np.zeros((m,n))
    for k in range(K):
        uk = u[:,k].reshape(m,1)
        vk = v[k].reshape(1,n)
        a += sigma[k] * np.dot(uk,vk)
    a[a < 0] = 0
    a[a > 255] = 255
    return np.rint(a).astype('uint8')

if __name__=="__main__":
    # 读取图片文件
    A = Image.open('lena.png','r')
    # 输出结果路径
    output_path = r'.\Pic'
    # 判断路径是否存在，若不存在进行创建
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    a = np.array(A)
    K = 30
    u_r,sigma_r,v_r = np.linalg.svd(a[:,:,0])
    u_g,sigma_g,v_g = np.linalg.svd(a[:,:,1])
    u_b,sigma_b,v_b = np.linalg.svd(a[:,:,2])
    plt.figure(figsize=(10,10),facecolor='w')
    mpl.rcParams['font.sans-serif'] = [u'simHei']
    mpl.rcParams['axes.unicode_minus'] = False
    for k in range(1,K+1):
        R = restore1(sigma_r,u_r,v_r,k)
        G = restore1(sigma_g,u_g,v_g,k)
        B = restore1(sigma_b,u_b,v_b,k)
        I = np.stack((R,G,B),axis=2)
        Image.fromarray(I).save('%s\\svd_%d.png' % (output_path,k))
        if k <= 30:
            plt.subplot(6,5,k)
            plt.imshow(I)
            plt.axis('off')
            plt.title(u'奇异值个数：%d' % k)
    plt.suptitle(u'SVD与图像分解',fontsize=20)
    plt.tight_layout(0.3,rect=(0,0,1,0.92))
    plt.subplots_adjust(top=0.9)
    plt.show()