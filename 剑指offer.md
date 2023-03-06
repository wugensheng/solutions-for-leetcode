  

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



## 第 2 天——链表（简单）链表



#### [剑指 Offer 06. 从尾到头打印链表](https://leetcode.cn/problems/cong-wei-dao-tou-da-yin-lian-biao-lcof/)

```` java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) { val = x; }
 * }
 */
class Solution {
    // public int[] reversePrint(ListNode head) { 单纯的数组操作最快
    //     // List<Integer> list = new ArrayList<>();
    //     // List<Integer> list = new LinkedList<>();
    //     int[] list = new int[10000];
    //     ListNode cur = head;
    //     int index = 0;
    //     while (cur != null) {
    //         list[index++] = cur.val;
    //         cur = cur.next;
    //     }

    //     int[] res = new int[index];
    //     int temp = index;
    //     for (int i = 0; i < index; i++) {
    //         res[i] = list[--temp];
    //     }

    //     return res;
    // }

    public int[] reversePrint(ListNode head) {
        Stack<Integer> st = new Stack<>();
        int index = 0;
        ListNode cur = head;
        while (cur != null) {
            st.push(cur.val);
            index++;
            cur = cur.next;
        }

        int[] res = new int[index];
        int i = 0;
        while (!st.isEmpty()) {
            res[i++] = st.pop();
        }

        return res;
    }
}
````



#### [剑指 Offer 24. 反转链表](https://leetcode.cn/problems/fan-zhuan-lian-biao-lcof/)

``` java
class Solution {
    public ListNode reverseList(ListNode head) {
        if (head == null) return head;
        ListNode cur = head, pre = null, temp = null;
 
        while (cur.next != null) {
            temp = cur.next;
            cur.next = pre;
            pre = cur;
            cur = temp;
        }

        cur.next = pre;
        return cur;
    }
}
```



#### [剑指 Offer 35. 复杂链表的复制](https://leetcode.cn/problems/fu-za-lian-biao-de-fu-zhi-lcof/)

``` java
class Solution {
    public Node copyRandomList(Node head) {
        if (head == null) return null;
        Map<Object, Integer> originMap = new HashMap<>(); // 记录每个节点和指针的映射关系
        Map<Integer, Object> copyMap = new HashMap<>(); // copy和origin之间只有下标能对应上，所以要进行两次映射

        Node originCur = head;
        int index = 0;
        Node resNode = new Node(head.val);
        Node copyCur = resNode;
        while (originCur != null) {
            originMap.put(originCur, index);
            copyMap.put(index, copyCur);
            index++;
            originCur = originCur.next;
            if(originCur != null) copyCur.next = new Node(originCur.val);
            else copyCur.next = null;
            copyCur = copyCur.next;
        }

        originCur = head;
        copyCur = resNode;
        while (originCur != null) {
            if (originCur.random == null) {
                copyCur.random = null;
            } else {
                copyCur.random = (Node)copyMap.get(originMap.get(originCur.random));
            }
            copyCur = copyCur.next;
            originCur = originCur.next;
        }

        return resNode;
    }
}
```



## 第3天——字符串（简单）

#### [剑指 Offer 05. 替换空格](https://leetcode.cn/problems/ti-huan-kong-ge-lcof/)

```` java
class Solution {
    public String replaceSpace(String s) {
        StringBuilder sb = new StringBuilder();
        char[] ss = s.toCharArray();
        for (int i = 0; i < ss.length; i++) {
            if (ss[i] == ' ') {
                sb.append("%20");
            } else {
                sb.append(ss[i]);
            }
        }

        return sb.toString();
    }
}
````



#### [剑指 Offer 58 - II. 左旋转字符串](https://leetcode.cn/problems/zuo-xuan-zhuan-zi-fu-chuan-lcof/)

``` java
class Solution {
    public String reverseLeftWords(String s, int n) {
        char[] ss = s.toCharArray();
        reverseWords(ss, 0, n - 1);
        reverseWords(ss, n, ss.length - 1);
        reverseWords(ss, 0, ss.length  - 1);
        return new String(ss, 0, ss.length);
    }

    public void reverseWords(char[] s, int start, int end) {
        while (start <= end) {
            char temp = s[start];
            s[start] = s[end];
            s[end] = temp;
            start++;
            end--;
        }
    }
}
```



## 第4天——查找算法（简单）

#### [剑指 Offer 03. 数组中重复的数字](https://leetcode.cn/problems/shu-zu-zhong-zhong-fu-de-shu-zi-lcof/)

``` java
class Solution {
    public int findRepeatNumber(int[] nums) { // 常规做法
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {
            if (map.containsKey(nums[i])) {
                return nums[i];
            } else {
                map.put(nums[i], 1);
            }
        }

        return -1;
    }
}
```

``` java
class Solution {
    public int findRepeatNumber(int[] nums) { // 将元素索引与元素值建立一对多的映射关系
        int i = 0; 
        while (i < nums.length) {
            if (nums[i] == i) {
                i++;
                continue;
            }
            if (nums[nums[i]] == nums[i]) return nums[i];
            int temp = nums[i];
            nums[i] = nums[temp];
            nums[temp] = temp;
        }

        return -1;
    }
}
```



#### [剑指 Offer 53 - I. 在排序数组中查找数字 I](https://leetcode.cn/problems/zai-pai-xu-shu-zu-zhong-cha-zhao-shu-zi-lcof/)

```` java
class Solution {
    public int search(int[] nums, int target) {
        if (nums.length == 0 || target > nums[nums.length - 1] || target < nums[0]) return 0;
        int leftBoard = -1, rightBoard = -1;
        int left = 0, right = nums.length - 1;
        while (left <= right) {
            int mid = left + ((right - left) >> 1);
            if (nums[mid] < target) left = mid + 1;
            else {
                right = mid - 1;
                leftBoard = right;
            }
        }

        left = 0;
        right = nums.length - 1;
        while (left <= right) {
            int mid = left + ((right - left) >> 1);
            if (nums[mid] > target) right = mid - 1;
            else {
                left = mid + 1;
                rightBoard = left;
            }
        }

        if (rightBoard > leftBoard + 1) return rightBoard - leftBoard - 1;
        return 0;
    }
}
````



#### [剑指 Offer 53 - II. 0～n-1中缺失的数字](https://leetcode.cn/problems/que-shi-de-shu-zi-lcof/)

``` java
class Solution {
    public int missingNumber(int[] nums) {
        int left = 0, right = nums.length - 1;
        int leftBoard = nums.length - 1; // 当最后一个缺省时， leftboard 得不到更新
        while (left <= right) {
            int mid = left + ((right - left) >> 1);
            if (nums[mid] == mid) left = mid + 1;
            else {
                right = mid - 1;
                leftBoard = right;
            }
        }
        return leftBoard + 1;
    }
}
```

