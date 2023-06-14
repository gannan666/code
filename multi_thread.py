import threading
import requests
import time

x = []
y = []

def craw(url):
    r = requests.get(url)
    num = len(r.text)  # 爬取博客园当页的文字数
    print("bn iusb", num)
    return num  # 返回当页文字数

class MyThread(threading.Thread):  # 重写threading.Thread类，加入获取返回值的函数

    def __init__(self, line):
        threading.Thread.__init__(self)
        self.line = line                # 初始化传入的line

    def run(self):                    # 新加入的函数，该函数目的：
        self.result = craw(self.line)  # ①调craw(arg)函数，并将初试化的url以参数传递
                                      # ②获取craw(arg)函数的返回值存入本类的定义的值result中

    def get_result(self):  # 新加入函数，该函数目的：返回run()函数得到的result
        return self.result

def multi_thread():
    print("start")
    threads = []           # 定义一个线程组
    for line in lines:
        threads.append(    # 线程组中加入赋值后的MyThread类
            MyThread(line)  # 将每一个url传到重写的MyThread类中
        )
    for thread in threads: # 每个线程组start
        thread.start()

    for thread in threads: # 每个线程组join
        thread.join()

    list = []
    for thread in threads:
        x_, y_ = thread.get_result()  # 每个线程返回结果(result)加入列表中
        x.append(x_)
        y.append(y_)
    print("end")
    return list  # 返回多线程返回的结果组成的列表

if __name__ == '__main__':
    lines = ["https://blog.csdn.net/csdnnews/article/details/131098677", "https://blog.csdn.net/fengbingchun/article/details/131024200"]
    start_time = time.time()
    result_multi = multi_thread()
    print(result_multi)
    end_time = time.time()