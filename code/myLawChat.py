import tkinter as tk
import tkinter.messagebox
import datetime
import os
import threading
import time
from classiry_similarity2 import jiaohu
from classiry_similarity2 import getClass
from match import law

'''
   机器人回答类型状态码 reType

   0 默认提示
   1 回答输入问题的type
   2 回答输入问题的答案
   3 回答输入问题的匹配法条
   4 问题选项
'''

def analyze_and_build(name,data,reType):
    timenow = datetime.datetime.now().timestamp()
    time=datetime.datetime.fromtimestamp(float(timenow)).strftime('%Y-%m-%d %H:%M:%S')
    if reType==0:
        content=data
        return '%s\n%s : %s\n\n' % (time,name,content)
    if reType==1:
        content=data
        return '%s\n%s : 您的问题属于%s \n\n' % (time,name,content)
    if reType==2:
        content=data
        return '%s\n%s : 为您找到最相关的问题与回答:\n1、%s\n解答:%s\n\n2、%s\n解答:%s\n\n3、%s\n解答:%s\n\n ' % (time,name,content[0][1],content[0][2],content[1][1],content[1][2],content[3][1],content[3][2])
    if reType==3:
        content=data
        return '%s\n%s : 为您匹配到如下法条:\n1、%s\n\n2、%s\n\n3、%s\n\n' % (time,name,content[0],content[1],content[2])
    if reType==4:
        content=data
        return '%s\n%s : \n%s\n%s\n%s\n\n' % (time,name,data[0],data[1],data[2])
    




qdata = ''
flag = 0

def mainview():
    name=nameEntry.get()
    choice  = ''
    flag = 0


 
    if name=='' :
        tkinter.messagebox.showinfo(title='login failed',message='Value cannot be empty!')
    else:
        loginWindow.destroy()
        mainWindow=tk.Tk()
        mainWindow.title('法律智能问答系统(%s)' % name)
        mainWindow.geometry('450x400')

        showText=tk.Text(mainWindow,height=20,width=60)
        emptyLabel=tk.Label(mainWindow,height=1)
        writeText=tk.Text(mainWindow,height=4,width=60)
        showText.tag_config('red',foreground='red')
        showText.tag_config('default',background='yellow',foreground='black')

        showText.insert('end','**********************法律智能问答系统**********************\n')
        Robot = '智能机器管家'
        welcome = '欢迎来到法律智能问答系统，请输入你想咨询的法律问题'
        showText.insert('end',analyze_and_build(Robot,welcome,0),'default')


        showText.pack()
        emptyLabel.pack()
        writeText.pack()


        def Response(data):
            global flag
            if(flag == 0):
                global qdata
                qdata = data.strip()
                flag = 1
            
            choice = data.strip()
            if choice=='1':
                nClass = getClass(qdata)
                lawResult = law(qdata,nClass)
                showText.insert('end',analyze_and_build(Robot,lawResult,3))  
            elif choice=='2':
                moreQ = '请输入你想咨询的法律问题'
                showText.insert('end',analyze_and_build(Robot,moreQ,0),'default') 
                flag =0
            elif choice=='3':
                showText.insert('end','\n**********************正在退出问答系统**********************\n\n\n')
                content = '谢谢您的使用'
                showText.insert('end',analyze_and_build(Robot,content,0),'default')    
            else:
                class_name,QAList = jiaohu(data)
                showText.insert('end',analyze_and_build(Robot,class_name,1))
                showText.insert('end',analyze_and_build(Robot,QAList,2))
                choiceStr = ['是否为您返回相关法条----------1',
                             '是否开始查询下一个问题--------2',
                             '任何时候结束程序--------------3']
                showText.insert('end',analyze_and_build(Robot,choiceStr,4),'default')

        def send_message():
            data=writeText.get('0.0','end')
            if data=='/n' or data=='':
                pass
            else:
                showText.insert('end',analyze_and_build(name,data,0),'red')
                writeText.delete('0.0','end') 
                Response(data)

                

            

        sendButton=tk.Button(mainWindow,text='send',width=5,height=1,command=send_message)

        sendButton.pack()




        mainWindow.protocol('WM_DELETE_WINDOW',quit)
        mainWindow.mainloop()






loginWindow=tk.Tk()
loginWindow.title('login')
loginWindow.geometry('200x250')

loginLabel=tk.Label(loginWindow,
    text='Login',
    font=('Arial',15),
    width=10,height=2)

nameLabel=tk.Label(loginWindow,
    text='name:',
    font=('Arial',10),
    width=6,height=1)


nameEntry=tk.Entry(loginWindow)
loginButton=tk.Button(loginWindow,
    text='login',
    width=10,
    height=1,
    command=mainview)


loginLabel.pack()
nameLabel.pack()
nameEntry.pack()
loginButton.pack()


loginWindow.mainloop()
