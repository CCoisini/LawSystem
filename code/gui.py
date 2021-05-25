import tkinter as tk
import tkinter.messagebox
import socket
import datetime
import os
import threading
import time

def analyze_and_build(raw):
    rawlist=raw.decode('utf-8').split('-$-')
    method=rawlist[0]
    if method=='n':
        time=datetime.datetime.fromtimestamp(float(rawlist[1])).strftime('%Y-%m-%d %H:%M:%S')
        name=rawlist[2]
        content=rawlist[3]
        return '%s\n%s : %s\n\n' % (time,name,content)
    elif method=='e':
        time=datetime.datetime.fromtimestamp(float(rawlist[1])).strftime('%Y-%m-%d %H:%M:%S')
        name=rawlist[2]
        return '%s\n%s加入了聊天\n\n' % (time,name)
    elif method=='q':
        name=rawlist[1]
        return '%s退出了聊天\n\n' % name

def mainview():
    name=nameEntry.get()
    host=hostEntry.get()
    port=portEntry.get()
    tkinter.messagebox.showinfo(title='欢迎来到法律智能问答系统',message='请输入你想要查询的法律问题')
    if name=='' or host=='' or port=='':
        tkinter.messagebox.showinfo(title='login failed',message='Value cannot be empty!')
    else:
        s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        s.connect((host,int(port)))
        loginWindow.destroy()
        mainWindow=tk.Tk()
        mainWindow.title('Chatroom(%s)' % name)
        mainWindow.geometry('440x390')

        showText=tk.Text(mainWindow,height=20,width=60)
        emptyLabel=tk.Label(mainWindow,height=1)
        writeText=tk.Text(mainWindow,height=4,width=60)
        showText.tag_config('red',foreground='red')


        showText.pack()
        emptyLabel.pack()
        writeText.pack()


        s.send(('e-$-%s-$-%s' % (datetime.datetime.now().timestamp(),name)).encode('utf-8'))

        def fresh():
            while True:
                data=s.recv(1024)
                rawlist=data.decode('utf-8').split('-$-')
                if (rawlist[0]=='n' or rawlist[0]=='e') and rawlist[2]==name:
                    showText.insert('end',analyze_and_build(data),'red')
                else:
                    showText.insert('end',analyze_and_build(data))
                showText.see('end')

        t1=threading.Thread(target=fresh)
        t1.start()

        def send_message():
            data=writeText.get('0.0','end')
            if data=='\n':
                pass
            else:
                s.send(('n-$-%s-$-%s-$-%s' % (datetime.datetime.now().timestamp(),name,data)).encode('utf-8'))
                writeText.delete('0.0','end')

        sendButton=tk.Button(mainWindow,text='send',width=5,height=1,command=send_message)
        sendButton.pack()

        def quit():
            s.send(('q-$-%s' % name).encode('utf-8'))
            time.sleep(0.5)
            s.send('_quitchatroom'.encode('utf-8'))
            mainWindow.destroy()
            os._exit(0)

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

hostLabel=tk.Label(loginWindow,
    text='host:',
    font=('Arial',10),
    width=6,height=1)

portLabel=tk.Label(loginWindow,
    text='port:',
    font=('Arial',10),
    width=6,height=1)

nameEntry=tk.Entry(loginWindow)
hostEntry=tk.Entry(loginWindow)
portEntry=tk.Entry(loginWindow)

loginButton=tk.Button(loginWindow,
    text='login',
    width=10,
    height=1,
    command=mainview)

loginLabel.pack()
nameLabel.pack()
nameEntry.pack()
hostLabel.pack()
hostEntry.pack()
portLabel.pack()
portEntry.pack()
loginButton.pack()

loginWindow.mainloop()