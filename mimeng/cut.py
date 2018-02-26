import jieba
import jieba.posseg
import sys

for i in range(0,50):
    f = open(str(i)+".txt","r",encoding= 'gbk')
    g = open("mm"+str(i) + "f.txt", 'w',encoding= 'utf-8')
    h = open("mm"+str(i) + "g.txt", 'w',encoding= 'utf-8')
    lines=[]
    for line in f:
        lines.append(line.strip())
    f.close()
    str1=" ".join(lines)
    seg=jieba.posseg.cut(str1)
    punc = open("punctuation.txt",'r')
    pr = punc.read()
    #prd = pr.decode('gbk')
    p = pr.split()
    lreal = []
    lvir = []
    punclist = ["，",",","“","”","‘","’",".","。",":","：",";","；","！",
                "!","？","?","（","）","(",")",'、','——','《','》','…',"……"]
    passlist = ['\\',"0","1","2","3","4","5","6","7","8","9",'/','-','$','#']
    puncreplace = ["逗号","逗号","引号","引号","引号","引号","句号","句号","冒号",
                   "冒号","分号","分号","感叹号","感叹号","问号","问号","括号",
                   "括号","括号","括号",'顿号','破折号','书名号','书名号','省略号',"省略号"]
    for i in seg:
        if i.word in passlist:
            pass
        elif i.word in punclist:
            for j in range(len(punclist)):
                if i.word == punclist[j]:
                    lvir.append(puncreplace[j])
        elif i.flag in ['n','v','a'] and i.word not in p:
            lreal.append(i.word)
        elif i.flag not in ['eng'] and i.word in p:
            lvir.append('填充'+i.word)
    strreal = " ".join(lreal)
    strvir = " ".join(lvir)

    g.write(strreal)
    g.close()

    h.write(strvir)
    h.close()
