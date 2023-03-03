  

## 第1天——栈与队列

#### [剑指 Offer 09. 用两个栈实现队列](https://leetcode.cn/problems/yong-liang-ge-zhan-shi-xian-dui-lie-lcof/)

两个栈一个用来入队列，一个用来出队列，当出队列栈为空时讲入队列栈弹入其中，可以保证先入先出

``` java
class CQueue {
    Stack<Integer> st1 = new Stack<>(); // 入队列
    Stack<Integer> st2 = new Stack<>(); // 出队列

    public CQueue() {
        
    }
    
    public void appendTail(int value) {
        st1.push(value);
    }
    
    public int deleteHead() {
        if (!st2.isEmpty()) {
            return st2.pop();
        }
        if (!st1.isEmpty()) {
            while (!st1.isEmpty()) st2.push(st1.pop());
            return st2.pop();
        }

        return -1;
    }
}

/**
 * Your CQueue object will be instantiated and called as such:
 * CQueue obj = new CQueue();
 * obj.appendTail(value);
 * int param_2 = obj.deleteHead();
 */
```

#### [剑指 Offer 30. 包含min函数的栈](https://leetcode.cn/problems/bao-han-minhan-shu-de-zhan-lcof/)

``` java
class MinStack {

    Stack<Integer> stack = new Stack<>(); // 真实栈
    Stack<Integer> minStack = new Stack<>(); // 辅助栈, 用来记录每个值入栈时刻snapshot的最小值，并同步入栈

    /** initialize your data structure here. */
    public MinStack() {

    }
    
    public void push(int x) {
        stack.push(x);
        if (minStack.isEmpty()) {
            minStack.push(x);
        } else {
            minStack.push(minStack.peek() > x ? x : minStack.peek());
        }
    }
    
    public void pop() {
        stack.pop();
        minStack.pop();
    }
    
    public int top() {
        return stack.peek();
    }
    
    public int min() {
        return minStack.peek();
    }
}

/**
 * Your MinStack object will be instantiated and called as such:
 * MinStack obj = new MinStack();
 * obj.push(x);
 * obj.pop();
 * int param_3 = obj.top();
 * int param_4 = obj.min();
 */
```



## 第 2 天链表（简单）链表



