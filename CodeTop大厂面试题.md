

####  [3. 无重复字符的最长子串](https://leetcode.cn/problems/longest-substring-without-repeating-characters/)

用滑动窗口/双指针维护一个不重复序列：

- 窗口右侧不重复则加入窗口
- 窗口右侧重复则收缩窗口左侧，直至窗口不重复
- 记录窗口最大长度

```java
class Solution {
    // 双指针 维护一个不重复子串
    public int lengthOfLongestSubstring(String s) {
        char[] ss = s.toCharArray();
        if (ss.length <= 1) return ss.length;

        int[] mark = new int[256];
        int maxLength = 0;
        int left = 0, right = 0;

        while (right < ss.length) {
            if (mark[ss[right]] == 0) {
                mark[ss[right]] = 1;
                if (right - left + 1 > maxLength) maxLength = right - left + 1;
                right++;
                continue;
            }

            while (ss[left] != ss[right]) {
                mark[ss[left]] = 0;
                left++;
            }
            left++;
            right++;
        }
        return maxLength;
    }
}
```



#### [25. K 个一组翻转链表](https://leetcode.cn/problems/reverse-nodes-in-k-group/)

``` java
class Solution {
    // 对翻转操作进行封装
    public ListNode reverseKGroup(ListNode head, int k) {
        ListNode danny = new ListNode();
        danny.next = head;
        // start指向翻转部分的第一个节点，end指向翻转部分的最后一个节点
        ListNode start = head, end = danny, next = null, pre = danny;
        // 当start不为空的时候下一轮翻转节点继续遍历
        while (start != null) {
            int i = k;
            while (i > 0) {
                i--;
                end = end.next;
                if (end == null) return danny.next;
            }
            // 记录下次开始的节点
            next = end.next;
            // 翻转
            reverse(start, end);
            // 将翻转完的链表链接上
            start.next = next;
            pre.next = end;
            // 更新
            pre = start;
            end = start;
            start = pre.next;
        }
        return danny.next;
    }

    // 对指定节点之间的链表进行翻转
    public void reverse(ListNode start, ListNode end) {
        ListNode pre = null, cur = start, temp = null;

        while (cur != end) {
            temp = cur.next;
            cur.next = pre;
            pre = cur;
            cur = temp;
        }
        cur.next = pre;
    }
}
```



#### [206. 反转链表](https://leetcode.cn/problems/reverse-linked-list/)

``` java
class Solution {
    // 指定区间的链表的翻转操作
    public ListNode reverseList(ListNode head) {
        if (head == null || head.next == null) return head;

        // pre指向已经翻转过的链表的最后一个部分，cur指向当前要翻转的链表的开始部分
        ListNode pre = null, cur = head;
        
        while (cur != null) {
            ListNode temp = cur.next;
            cur.next = pre;
            pre = cur;
            cur = temp;
        }

        return pre;
    }
}
```



#### [146. LRU 缓存](https://leetcode.cn/problems/lru-cache/)

哈希表+双向链表

``` java
class LRUCache {
    // LRU: Least Recently Used 是一种常见的缓存淘汰算法，淘汰最久没有被使用过的数据
    // 使用哈希表和双向链表实现
    // 哈希表用来定位关键字
    // 双向链表用来对数据进行排序，最近使用过的放在最前面
  	// hashmap获取关键字对应的变量
  	// 双向链表可以实现，常数时间复杂度的插入删除操作，以及移动

    class DLinkedNode {
        int key;
        int value;
        DLinkedNode pre;
        DLinkedNode next;
        public DLinkedNode() {}
        public DLinkedNode(int key, int value) {
            this.key = key;
            this.value = value;
        }
    }

    private Map<Integer, DLinkedNode> cache = new HashMap<Integer, DLinkedNode>();
    private int capacity;
    private int size;
    private DLinkedNode head, tail; // 虚拟头尾节点


    public LRUCache(int capacity) {
        this.capacity = capacity;
        this.size = 0;
        head = new DLinkedNode();
        tail = new DLinkedNode();
        head.next = tail;
        tail.pre = head;
    }

    public int get(int key) {
        DLinkedNode node = cache.get(key);
        if (node == null) return -1;
        moveToHead(node);
        return node.value;
    }

    public void put(int key, int value) {
        DLinkedNode node = cache.get(key);
        if (node == null) {
            node = new DLinkedNode(key, value);
            cache.put(key, node);
            addToHead(node);
            size++;
            if (size > capacity) {
                cache.remove(tail.pre.key);
                removeNode(tail.pre);
                size--;
            }
        } else {
            node.value = value;
            moveToHead(node);
        }
    }

    public void moveToHead(DLinkedNode node) {
        removeNode(node);
        addToHead(node);
    }

    public void removeNode(DLinkedNode node) {
        DLinkedNode pre = node.pre, next = node.next;
        pre.next = next;
        next.pre = pre;
    }

    public void addToHead(DLinkedNode node) {
        DLinkedNode next = head.next;
        node.pre = head;
        node.next = next;
        head.next = node;
        next.pre = node;
    }
}
```



#### [215. 数组中的第K个最大元素](https://leetcode.cn/problems/kth-largest-element-in-an-array/)

``` java
class Solution {
    // 快速排序 只对需要排序的部分进一步排序，期望时间复杂度为O(n)
    public int findKthLargest(int[] nums, int k) {
        return quickSelect(nums, 0, nums.length - 1, nums.length - k);
    }

    // 1. 随机选择基元，然后划分数组
    // 2. 判断划分后的基元与目标的相对位置，有选择的递归排序
    public int quickSelect(int[] nums, int l, int r, int index) { // index 是倒数第k个元素的下标
        int pos = randomPartition(nums, l, r);
        if (pos == index) return nums[pos]; // 找到第k大的元素直接退出
        else {
            // 如果index在pos的右边，则只需要对右边进行递归排序，如果在左边就对左边进行递归排序
            return index > pos ? quickSelect(nums, pos + 1, r, index) : quickSelect(nums, l, pos - 1, index);
        }
    }

    // 该函数随机选择基元，然后划分数组，并返回基元的下标
    public int randomPartition(int[] nums, int l, int r) {
        int i = new Random().nextInt(r - l + 1) + l;
        swap(nums, i, r);
        return partition(nums, l, r);
    }

    public int partition(int[] nums, int l, int r) {
        int flag = nums[r];
        int i = l - 1;
        for (int j = l; j < r; j++) {
            if (nums[j] < flag) {
                i++;
                swap(nums, i, j);
            }
        }
        swap(nums, i + 1, r);
        return i + 1;
    }

    public void swap(int[] nums, int i, int j) {
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }
}
```



#### [912. 排序数组](https://leetcode.cn/problems/sort-an-array/)

``` java
class Solution {
    public int[] sortArray(int[] nums) {
        quickSort(nums, 0, nums.length - 1);
        return nums;
    }

    public void quickSort(int[] nums, int l, int r) {
        if (l < r) {
            int pos = randomPartition(nums, l, r);
            quickSort(nums, l, pos - 1);
            quickSort(nums, pos + 1, r);
        }
    }

    public int randomPartition(int[] nums, int l, int r) {
        int i = new Random().nextInt(r - l + 1) + l;
        swap(nums, i, r);
        return partition(nums, l, r);
    }

    public int partition(int[] nums, int l, int r) {
        int flag = nums[r];
        int i = l - 1; // i是小于flag的数的边界
        for (int j = l; j <= r - 1; j++) {
            if (nums[j] <= flag) { // nums[j] 是左边的数, 从小到大排序
                i++;
                swap(nums, i, j); // 将边界右边的数和小于边界的数置换
            }
        }
        swap(nums, i + 1, r);
        return i + 1;
    }

    public void swap(int[] nums, int i, int j) {
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }
}
```



#### [53. 最大子数组和](https://leetcode.cn/problems/maximum-subarray/)

动态规划

``` java
class Solution {
    public int maxSubArray(int[] nums) {
        int[] dp = new int[nums.length + 1];
        dp[0] = 0;
        int res = Integer.MIN_VALUE;
        for (int i = 1; i <= nums.length; i++) {
            dp[i] = Math.max(dp[i - 1] + nums[i - 1], nums[i - 1]);
            if (dp[i] > res) res = dp[i];
        }

        return res;
    }
}
```

分治算法



#### [15. 三数之和](https://leetcode.cn/problems/3sum/)

``` java
class Solution {
    public List<List<Integer>> threeSum(int[] nums) {
        Arrays.sort(nums);
        List<List<Integer>> res = new ArrayList<>();

        for (int i = 0; i < nums.length; i++) { // 枚举a
            if (nums[i] > 0) break;
            if (i > 0 && nums[i] == nums[i - 1]) continue; // 对a去重

            int left = i + 1, right = nums.length - 1; // 枚举b和c
            while (left < right) {
                int sum = nums[i] + nums[left] + nums[right];
                if (sum > 0) right--;
                else if (sum < 0) left++;
                else {
                    List<Integer> list = new ArrayList<>();
                    list.add(nums[i]);
                    list.add(nums[left]);
                    list.add(nums[right]);
                    res.add(list);

                    while (left < right && nums[left] == nums[left + 1]) left++; // 对b去重
                    while (left < right && nums[right] == nums[right - 1]) right--; // 对c去重
                    left++;
                    right--;
                }
            }
        }

        return res;
    }
}
```



#### [21. 合并两个有序链表](https://leetcode.cn/problems/merge-two-sorted-lists/)

``` java
class Solution {
    public ListNode mergeTwoLists(ListNode list1, ListNode list2) {
        ListNode dummy = new ListNode();
        ListNode cur = dummy;

        while (list1 != null && list2 != null) {
            if (list1.val < list2.val) {
                cur.next = list1;
                list1 = list1.next;
            } else {
                cur.next = list2;
                list2 = list2.next;
            }
            cur = cur.next;
        }

        if (list1 != null) cur.next = list1;
        if (list2 != null) cur.next = list2;

        return dummy.next;
    }
}
```



#### [1. 两数之和](https://leetcode.cn/problems/two-sum/)

``` java
class Solution {
    public int[] twoSum(int[] nums, int target) {
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {
            if (map.containsKey(target - nums[i])) {
                return new int[]{i, map.get(target - nums[i])};
            }
            map.put(nums[i], i);
        }
        return new int[]{0, 0};
    }
}
```



#### [102. 二叉树的层序遍历](https://leetcode.cn/problems/binary-tree-level-order-traversal/)

``` java
class Solution {
    public List<List<Integer>> levelOrder(TreeNode root) {
        List<List<Integer>> res = new ArrayList();
        if (root == null) return res;

        // 当队列中某一层有节点时，将该层节点全部取出，同时将下层节点放入
        Queue<TreeNode> q = new LinkedList();
        q.offer(root);
        while (!q.isEmpty()) {
            int size = q.size();
            List<Integer> list = new ArrayList();
            for (int i = 0; i < size; i++) {
                TreeNode node = q.poll();
                list.add(node.val);
                if (node.left != null) q.offer(node.left);
                if (node.right != null) q.offer(node.right);
            }
            res.add(list);
        }
        
        return res;
    }
}
```



#### [33. 搜索旋转排序数组](https://leetcode.cn/problems/search-in-rotated-sorted-array/)√

[题解](https://leetcode.cn/problems/search-in-rotated-sorted-array/solution/er-fen-cha-zhao-di-gui-bian-xing-xuan-zh-j9cg/)

``` java
class Solution {
    public int search(int[] nums, int target) {
        return binarySearch(nums, 0, nums.length - 1, target);
    }

    public int binarySearch(int[] nums, int l, int r, int target) {
        if (l > r) return -1;
        int mid = (l + r + 1) / 2;
        if (nums[mid] == target) return mid;
        else {
            if (nums[mid] >= nums[l]) { // 等与号不要忘
                if (nums[l] <= target && target < nums[mid]) return binarySearch(nums, l, mid - 1, target);
                else return binarySearch(nums, mid + 1, r, target);
            }
            if (nums[mid] <= nums[r]) { // 等与号
                if (nums[mid] < target && nums[r] >= target) return binarySearch(nums, mid + 1, r, target);
                else return binarySearch(nums, l, mid - 1, target);
            }
        }
        return -1;
    }
}
```



#### [20. 有效的括号](https://leetcode.cn/problems/valid-parentheses/)

``` java
class Solution {
    public boolean isValid(String s) {
        Deque<Character> st = new LinkedList<>();
        char[] ss = s.toCharArray();

        for (int i = 0; i < ss.length; i++) {
            if (ss[i] == '(' || ss[i] == '{' || ss[i] == '[') {
                st.push(ss[i]);
            } else {
                if (st.isEmpty()) return false;
                char temp = st.pop();
                if ((ss[i] == '}' && temp == '{') || (ss[i] == ']' && temp == '[') || (ss[i] == ')' && temp == '(')) continue; // 注意顺序
                return false;
            }
        }

        if (st.isEmpty()) return true;
        return false;
    }
}
```



#### [5. 最长回文子串](https://leetcode.cn/problems/longest-palindromic-substring/)

``` java
class Solution {
    public String longestPalindrome(String ss) {
        // dp[i][j];最长回文序列长度
        char[] s = ss.toCharArray();
        if (s.length == 1) return ss;
        int[][] dp = new int[s.length + 1][s.length + 1];
        int resLeft = 0, resRight = 0, res = 1;

        for (int i = 1; i <= s.length; i++) {
            for (int j = 0; j + i - 1 < s.length; j++) {
                int l = j, r = j + i - 1;
                if (s[l] == s[r]) {
                    if (i <= 2) dp[l][r] = 1;
                    else {
                        dp[l][r] = dp[l + 1][r - 1];
                    }
                } else {
                    dp[l][r] = 0;
                }

                if (dp[l][r] == 1 && i >= res) {
                    resLeft = l;
                    resRight = r;
                    res = i;
                }
            }
        }
        return new String(s, resLeft, res); // 第三个参数是长度
    }
} 
```



#### [121. 买卖股票的最佳时机](https://leetcode.cn/problems/best-time-to-buy-and-sell-stock/)

```` java
class Solution {
    // 贪心
    public int maxProfit(int[] prices) {
        int low = prices[0], res = 0; // low表示到目前为止的最低价格
        for (int i = 0; i < prices.length; i++) {
            low = Math.min(prices[i], low);
            res = Math.max(res, prices[i] - low);
        }

        return res;
    }

    // dp
    public int maxProfit(int[] prices) {
        // dp[i][0]: 第i天持有股票的最大价值
        // dp[i][1]: 第i天不持有股票的最大价值
        if (prices.length == 1) return 0;
        int[][] dp = new int[prices.length + 1][2];
        
        dp[0][0] -= prices[0];
        dp[0][1] = 0;
        for (int i = 1; i < prices.length; i++) {
            dp[i][0] = Math.max(dp[i - 1][0], -prices[i]);
            dp[i][1] = Math.max(dp[i - 1][1], dp[i - 1][0] + prices[i]);
        }

        return dp[prices.length - 1][1];
    }
}
````



#### [141. 环形链表](https://leetcode.cn/problems/linked-list-cycle/)

``` java
public class Solution {
    public boolean hasCycle(ListNode head) {
        if (head == null) return false;
        ListNode slow = head, fast = head;
        while (fast != null && fast.next != null) {
            slow = slow.next;
            fast = fast.next.next;
            if (slow == fast) return true;
        }

        return false;
    }
}
```



#### [88. 合并两个有序数组](https://leetcode.cn/problems/merge-sorted-array/)

``` java
class Solution {
    public void merge(int[] nums1, int m, int[] nums2, int n) {
        int newIndex = m + n - 1, index1 = m - 1, index2 = n - 1;
        while (index1 >= 0 && index2 >= 0) {
            if (nums1[index1] > nums2[index2]) {
                nums1[newIndex--] = nums1[index1--];
            } else {
                nums1[newIndex--] = nums2[index2--];
            }
        }

        while (index2 >= 0) nums1[newIndex--] = nums2[index2--];
    }
}
```



#### [103. 二叉树的锯齿形层序遍历](https://leetcode.cn/problems/binary-tree-zigzag-level-order-traversal/)

``` java
class Solution {
    public List<List<Integer>> zigzagLevelOrder(TreeNode root) {
        List<List<Integer>> res = new ArrayList<>();
        if (root == null) return res;

        Queue<TreeNode> q = new LinkedList<>();
        boolean fromLeft = true;
        q.offer(root);
        while (!q.isEmpty()) {
            // 双端队列，每次从队列中拿出来放入双端队列的时候，进行翻转
            Deque<Integer> list = new LinkedList<>();
            int size = q.size();
            for (int i = 0; i < size; i++) {
                TreeNode node = q.poll();
                if (fromLeft) {
                    list.addLast(node.val);
                } else {
                    list.addFirst(node.val);
                }
                if (node.left != null) q.offer(node.left);
                if (node.right != null) q.offer(node.right);
            }
            fromLeft = !fromLeft;
            res.add(new ArrayList<>(list));
        }

        return res;
    }
}
```



#### [236. 二叉树的最近公共祖先](https://leetcode.cn/problems/lowest-common-ancestor-of-a-binary-tree/)

``` java
class Solution {
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if (root == null || root == p || root == q) return root;

        TreeNode left = lowestCommonAncestor(root.left, p, q);
        TreeNode right = lowestCommonAncestor(root.right, p, q);

        if (left == null) return right;
        if (right == null) return left;
        return root;
    }
}
```



#### [6. 全排列](https://leetcode.cn/problems/permutations/)

``` java
class Solution {
    List<List<Integer>> res = new ArrayList<>();
    List<Integer> path = new ArrayList<>();

    public List<List<Integer>> permute(int[] nums) {
        int[] used = new int[nums.length];
        backtracking(nums, used); // used标记数组
        return res;
    }

    public void backtracking(int[] nums, int[] used) {
        if (path.size() == nums.length) {
            res.add(new ArrayList<>(path));
            return;
        }

        for (int i = 0; i < nums.length; i++) {
            if (used[i] == 1) continue;
            used[i] = 1;
            path.add(nums[i]);
            backtracking(nums, used);
            used[i] = 0;
            path.remove(path.size() - 1);
        }
    }
}
```



#### [160. 相交链表](https://leetcode.cn/problems/intersection-of-two-linked-lists/)

``` java
public class Solution {
    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        int m = 0, n = 0;
        ListNode curA = headA, curB = headB;

        while (curA != null) {
            m++;
            curA = curA.next;
        }
        while (curB != null) {
            n++;
            curB = curB.next;
        }
				
      	// 优雅永不过时
        int diff = Math.abs(n - m);
        if (n > m) {
            curA = headB;
            curB = headA;
        } else {
            curA = headA;
            curB = headB;
        }

        while (diff > 0) {
            curA = curA.next;
            diff--;
        }

        while (curA != null && curB != null) {
            if (curA == curB) return curA;
            curA = curA.next;
            curB = curB.next;
        }

        return null;

    }
}
```



#### [54. 螺旋矩阵](https://leetcode.cn/problems/spiral-matrix/)

``` java
class Solution {
    public List<Integer> spiralOrder(int[][] matrix) {
        int h = matrix.length, w = matrix[0].length;
        List<Integer> res = new ArrayList<>();

        int x = 0, y = 0;
        int loop = Math.min(h, w) / 2;
        for (int i = 0; i < loop; i++) {
            
            x = i;
            y = i;
            for (; y < w - i - 1; y++) res.add(matrix[x][y]);

            for (; x < h - i - 1; x++) res.add(matrix[x][y]);

            for (; y > i; y--) res.add(matrix[x][y]);

            for (; x > i; x--) res.add(matrix[x][y]);
        }

        int small = Math.min(h, w); // 最小
        if (small % 2 == 0) return res;

        x = loop;
        y = loop;
        if (w > h) {
            for (; y < w - loop; y++) res.add(matrix[x][y]);
        } else {
            for (; x < h - loop; x++) res.add(matrix[x][y]);
        }

        return res;
    }
}
```



#### [23. 合并K个升序链表](https://leetcode.cn/problems/merge-k-sorted-lists/)

``` java
class Solution {
    
    public ListNode mergeKLists(ListNode[] lists) {
        if (lists.length == 0) return null;
        return mergeList(lists, 0, lists.length - 1);
    }

    // 1. 将区间分成两半，递归各自合并排序
    // 2. 将两个区间合并得到的链表再次合并得到最终的有序链表
    public ListNode mergeList(ListNode[] lists, int l, int r) {
        if (l == r) return lists[l];
        
        int mid = (l + r) / 2;
        ListNode list1 = mergeList(lists, l, mid);
        ListNode list2 = mergeList(lists, mid + 1, r);

        return merge(list1, list2);
    }

    // 合并两个链表
    public ListNode merge(ListNode list1, ListNode list2) {
        ListNode dummy = new ListNode();
        ListNode cur = dummy;

        while (list1 != null && list2 != null) {
            if (list1.val < list2.val) {
                cur.next = list1;
                list1 = list1.next;
            } else {
                cur.next = list2;
                list2 = list2.next;
            }
            cur = cur.next;
        }
        cur.next = (list1 != null) ? list1 : list2;

        return dummy.next;
    }
}
```



#### [92. 反转链表 II](https://leetcode.cn/problems/reverse-linked-list-ii/)

``` java
class Solution {
    public ListNode reverseBetween(ListNode head, int left, int right) {
        if (left == right) return head;

        ListNode dummy = new ListNode();
        dummy.next = head;
        ListNode cur = dummy, end = null;
        int des = left;
        while (des > 1) { // 找到左边节点的前一个
            cur = cur.next;
            des--;
        }
        end = cur;
        des = right - left + 1;
        while (des > 0) { // 找右边的节点
            end = end.next;
            des--;
        }
        ListNode next = end.next;

        ListNode[] nodes = reverse(cur.next, end);
        cur.next = nodes[0];
        nodes[1].next = next;

        return dummy.next;

    }

    public ListNode[] reverse(ListNode start, ListNode end) { // 翻转链表
        ListNode cur = start, pre = null, temp = null;
        while (cur != end) {
            temp = cur.next;
            cur.next = pre;
            pre = cur;
            cur = temp;
        }
        cur.next = pre;
        return new ListNode[]{end, start};
    }
}
```



#### [415. 字符串相加](https://leetcode.cn/problems/add-strings/)

``` java
class Solution {
    public String addStrings(String num1, String num2) {
        char[] s1 = num1.toCharArray(), s2 = num2.toCharArray();
        StringBuilder sb = new StringBuilder();
        int i = s1.length - 1, j = s2.length - 1, add = 0; // 两个指针，一个进位符

        while (i >= 0 || j >= 0 || add != 0) {
            int x = i >= 0 ? s1[i] - '0' : 0;
            int y = j >= 0 ? s2[j] - '0' : 0;
            int sum = x + y + add;
            sb.append(sum % 10);
            add = sum / 10;
            i--;
            j--;
        }
        sb.reverse(); // 翻转
        
        return sb.toString();
    }
}
```



#### [142. 环形链表 II](https://leetcode.cn/problems/linked-list-cycle-ii/)

``` java
public class Solution {
    // 快慢指针判断链表是否有环
    // 找链表中首次入环的节点
    public ListNode detectCycle(ListNode head) {
        if (head == null || head.next == null) return null;

        ListNode fast = head, slow = head;

        while (!(fast == null || fast.next == null)) {
            fast = fast.next.next;
            slow = slow.next;
            // 发现有环
            // 重新用一个指针从起点出发，再次相遇就是首次入环的节点
            if (fast == slow) {
                slow = head;
                while (slow != fast) {
                    slow = slow.next;
                    fast = fast.next;
                }
                return fast;
            }
        }
        return null;
    }
}
```



####  [300. 最长递增子序列](https://leetcode.cn/problems/longest-increasing-subsequence/)

``` java
class Solution {
    public int lengthOfLIS(int[] nums) {
        int[] dp = new int[nums.length];
        dp[0] = 1;
        int res = 1;
        for (int i = 1; i < nums.length; i++) {
            dp[i] = 1;
            for (int j = 0; j < i; j++) {
                if (nums[i] > nums[j]) {
                    dp[i] = Math.max(dp[i], dp[j] + 1);
                }
            }
            if (dp[i] > res) res = dp[i];
        }

        return res;
    }
}
```



#### [42. 接雨水](https://leetcode.cn/problems/trapping-rain-water/)

``` java
class Solution {
    public int trap(int[] height) {
        Deque<Integer> dq = new LinkedList<>();
        int res = 0;

        for (int i = 0; i < height.length; i++) {
            while (!dq.isEmpty() && height[i] > height[dq.peek()]) {
                int mid = dq.pop();
                if (!dq.isEmpty()) {
                    int h = Math.min(height[dq.peek()], height[i]) - height[mid]; // 求的是最小的那个
                    int w = i - dq.peek() - 1; // 减去dq.peek()
                    res += h * w;
                }
            }
            dq.push(i);
        }

        return res;
    }
}
```



#### [143. 重排链表](https://leetcode.cn/problems/reorder-list/)

``` java
class Solution {
    public void reorderList(ListNode head) {
        // 计算长度,拆分
        int length = 0;
        ListNode cur = head;
        
        while (cur != null) {
            length++;
            cur = cur.next;
        }
        if (length <= 2) return;

        int half = (length / 2 + 1 + length % 2);
        cur = head;
        while (half - 2 > 0) {
            half--;
            cur = cur.next;
        }
        // 翻转后半部分
        ListNode list2 = reverse(cur.next), list1 = head;
        cur.next = null;
        // 合并
        merge(list1, list2);
    }

    public void merge(ListNode list1, ListNode list2) {
        ListNode dummy = new ListNode();
        ListNode cur = dummy;

        while (list1 != null && list2 != null) {
            cur.next = list1;
            list1 = list1.next;
            cur = cur.next;
            cur.next = list2;
            list2 = list2.next;
            cur = cur.next;
        }
        cur.next = list1 != null ? list1 : list2;
    }

    public ListNode reverse(ListNode head) {
        ListNode pre = null, cur = head, temp = null;

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



#### [124. 二叉树中的最大路径和](https://leetcode.cn/problems/binary-tree-maximum-path-sum/)

``` java
class Solution {
    int res = Integer.MIN_VALUE;

    public int maxPathSum(TreeNode root) {
        int maxSum = traversal(root);
        return res;
    }

    public int traversal(TreeNode root) {
        if (root == null) return 0;

        int leftMax = Math.max(traversal(root.left), 0);
        int rightMax = Math.max(traversal(root.right), 0);

        int maxSum = leftMax + rightMax + root.val; // 该节点做根节点的最大路径和

        if (maxSum > res) res = maxSum;
        
        return root.val + Math.max(leftMax, rightMax); // 返回该节点不坐根节点的最大路径和
    }
}
```



#### [94. 二叉树的中序遍历](https://leetcode.cn/problems/binary-tree-inorder-traversal/)

```` java
class Solution {
    List<Integer> res = new ArrayList<>();
    public List<Integer> inorderTraversal(TreeNode root) {
        traversal(root);
        return res;
    }

    public void traversal(TreeNode root) {
        if (root == null) return;
        traversal(root.left);
        res.add(root.val);
        traversal(root.right);
    }
}
````



#### [72. 编辑距离](https://leetcode.cn/problems/edit-distance/)

``` java
class Solution {
    public int minDistance(String word1, String word2) {
        // dp[i][j]: 表示w1[i - 1], w2[j - 1]结尾的最小操作次数
        char[] w1 = word1.toCharArray(), w2 = word2.toCharArray();
        int[][] dp = new int[w1.length + 1][w2.length + 1];
        for (int i = 0; i <= w1.length; i++) dp[i][0] = i;
        for (int j = 0; j <= w2.length; j++) dp[0][j] = j;

        for (int i = 1; i <= w1.length; i++) {
            for (int j = 1; j <= w2.length; j++) {
                if (w1[i - 1] == w2[j - 1]) dp[i][j] = dp[i - 1][j - 1]; // 相等不用操作
                else {// 不等，可以，w1删除，w2删除,w1或者w2修改
                    dp[i][j] = Math.min(dp[i - 1][j] + 1, Math.min(dp[i][j - 1] + 1, dp[i - 1][j - 1] + 1));
                }
            }
        }

        return dp[w1.length][w2.length];
    }
}
```



#### [232. 用栈实现队列](https://leetcode.cn/problems/implement-queue-using-stacks/)

``` java
class MyQueue {

    Deque<Integer> st1 = new LinkedList<>(); // 入队列
    Deque<Integer> st2 = new LinkedList<>(); // 出队列

    public MyQueue() {

    }
    
    public void push(int x) {
        st1.push(x);
    }
    
    public int pop() {
        if (!st2.isEmpty()) return st2.pop();
        while (!st1.isEmpty()) {
            st2.push(st1.pop());
        }
        return st2.pop();
    }
    
    public int peek() {
        if (!st2.isEmpty()) return st2.peek();
        while (!st1.isEmpty()) {
            st2.push(st1.pop());
        }
        return st2.peek();
    }
    
    public boolean empty() {
        return st1.isEmpty() && st2.isEmpty();
    }
}
```



#### [704. 二分查找](https://leetcode.cn/problems/binary-search/)

``` java
class Solution {
    public int search(int[] nums, int target) {
        int left = 0, right = nums.length - 1;

        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] == target) return mid;
            else if (nums[mid] > target) right = mid - 1;
            else left = mid + 1;
        }

        return -1;
    }
}
```



#### [19. 删除链表的倒数第 N 个结点](https://leetcode.cn/problems/remove-nth-node-from-end-of-list/)

``` java
class Solution {
    // 快慢指针，虚拟头结点
    public ListNode removeNthFromEnd(ListNode head, int n) {
        if (head == null) return head;

        ListNode danny = new ListNode();
        danny.next = head;
				// 确定fast和slow的起始位置
        ListNode fast = danny, slow = danny;
				
        while (n > 0) {
            fast = fast.next;
            n--;
        }

        while (fast.next != null) {
            fast = fast.next;
            slow = slow.next;
        }

        slow.next = slow.next.next;

        return danny.next;
    }
}
```



#### [199. 二叉树的右视图](https://leetcode.cn/problems/binary-tree-right-side-view/)

``` java
class Solution {
    // 二叉树的层序遍历的变种
    public List<Integer> rightSideView(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        if (root == null) return res;

        Queue<TreeNode> q = new LinkedList<>();
        q.offer(root);
        while (!q.isEmpty()) {
            int size = q.size();
            for (int i = 0; i < size; i++) {
                TreeNode node = q.poll();
                // 只对每一层的最后一个节点进行收集
                if (i == size - 1) {
                    res.add(node.val);
                }
                if (node.left != null) q.offer(node.left);
                if (node.right != null) q.offer(node.right);
            }
        }

        return res;
    }
}
```



#### [56. 合并区间](https://leetcode.cn/problems/merge-intervals/)

``` java
class Solution {
    public int[][] merge(int[][] intervals) {
        List<int[]> res = new ArrayList<>();
        Arrays.sort(intervals, (a, b) -> {
            // 排序规则，按左区间升序排序，相同情况况下，按右区间降序排序
            if (a[0] == b[0]) return b[1] - a[1];
            return a[0] - b[0];
        });

        int start = intervals[0][0], end = intervals[0][1]; // 标记当前的区间范围
        for (int i = 1; i < intervals.length; i++) {
            if (intervals[i][0] > end) { // 有新区间
                res.add(new int[]{start, end});
                start = intervals[i][0];
                end = intervals[i][1];
            } else {
                if (intervals[i][1] > end) end = intervals[i][1];
            }
        }
        res.add(new int[]{start, end});

        return res.toArray(new int[res.size()][2]);
    }
}
```



#### [70. 爬楼梯](https://leetcode.cn/problems/climbing-stairs/)

``` java
class Solution {
    public int climbStairs(int n) {
        int[] dp = new int[n + 5];
        dp[1] = 1;
        dp[2] = 2;
        for (int i = 3; i <= n; i++) {
            dp[i] = dp[i - 1] + dp[i - 2];
        }
        return dp[n];
    }
}
```



#### [31. 下一个排列](https://leetcode.cn/problems/next-permutation/)

``` java
class Solution {
    public void nextPermutation(int[] nums) {
        if (nums.length == 1) return;
        int i = nums.length - 2;
        while (i >= 0 && nums[i] >= nums[i + 1]) i--; // 找第一个逆序对
        if (i >= 0) {
            int j = nums.length - 1;
            while (j >= 0 && nums[j] <= nums[i]) j--; // 找升序列中第一个大于nums[i]数
            int temp = nums[j];
            nums[j] = nums[i];
            nums[i] = temp;
        }

        // revserse
        int left = i + 1, right = nums.length - 1;
        while (left <= right) {
            int temp = nums[left];
            nums[left] = nums[right];
            nums[right] = temp;
            left++;
            right--;
        }
        return;
    }
}
```



#### [148. 排序链表](https://leetcode.cn/problems/sort-list/)

从上至下的递归归并排序

``` java
class Solution {
    // 链表归并排序
    public ListNode sortList(ListNode head) {
        return mergeSort(head, null);
    }

    // 1. 对链表进行拆分，各自递归排序
    // 2. 对链表进行合并
    public ListNode mergeSort(ListNode head, ListNode end) {
        // 递归的出口，需要注意
        if (head == null) return null;
        if (head == end) {
            // 链表排序，最终的每个部分都是一个单独的链表，要加null
            head.next = null;
            return head;
        }

        ListNode dummy = new ListNode();
        dummy.next = head;
        ListNode fast = dummy, slow = dummy;
        while (fast != end) {
            slow = slow.next;
            fast = fast.next;
            if (fast != end) fast = fast.next;
        }
        // list1先排序的话next就丢失了，可以先保存
        ListNode list2 = mergeSort(slow.next, end);
        ListNode list1 = mergeSort(head, slow);

        return merge(list1, list2);
    }

    public ListNode merge(ListNode list1, ListNode list2) {
        ListNode dummy = new ListNode();
        ListNode cur = dummy;

        while (list1 != null && list2 != null) {
            if (list1.val < list2.val) {
                cur.next = list1;
                list1 = list1.next;
            } else {
                cur.next = list2;
                list2 = list2.next;
            }
            cur = cur.next;
        }

        cur.next = (list1 != null ? list1 : list2);
        return dummy.next;
    }
}
```



#### [82. 删除排序链表中的重复元素 II](https://leetcode.cn/problems/remove-duplicates-from-sorted-list-ii/)

``` java
class Solution {
    public ListNode deleteDuplicates(ListNode head) {
        if (head == null || head.next == null) return head;
        ListNode dummy = new ListNode();
        dummy.next = head;
        ListNode pre = dummy, cur = head;

        // 两个指针
        // pre指向前面保存的节点的最后一个节点
        // cur指向当前要判断的节点
        // 循环结束条件：当前判断节点为空或者是最后一个节点
        while (cur != null && cur.next != null) {
            if (cur.val == cur.next.val) {
                int target = cur.val;
                while (cur != null && cur.val == target) cur = cur.next;
                pre.next = cur;
                continue;
            }
            pre = cur;
            cur = cur.next;
        }
        return dummy.next;
    }
}
```



#### [1143. 最长公共子序列](https://leetcode.cn/problems/longest-common-subsequence/)

``` java
class Solution {
    public int longestCommonSubsequence(String text1, String text2) {
        char[] t1 = text1.toCharArray(), t2 = text2.toCharArray();
        int[][] dp = new int[t1.length + 1][t2.length + 1];

        for (int i = 1; i <= t1.length; i++) {
            for (int j = 1; j <= t2.length; j++) {
                if (t1[i - 1] == t2[j - 1]) dp[i][j] = dp[i - 1][j - 1] + 1;
                else {
                    dp[i][j] = Math.max(Math.max(dp[i - 1][j], dp[i][j - 1]), dp[i - 1][j - 1]);
                }
            }
        }

        return dp[t1.length][t2.length];
    }
}
```





#### [69. x 的平方根 ](https://leetcode.cn/problems/sqrtx/)

```` java
class Solution {
    public int mySqrt(int x) {
        long left = 0, right = x;
        while (left <= right) {
            long mid = left + (right - left) / 2;
            if (mid * mid == x) return (int)mid;
            else if (mid * mid > x) right = mid - 1;
            else left = mid + 1;
        }

        return (int)right;
    }
}
````



#### [2. 两数相加](https://leetcode.cn/problems/add-two-numbers/)

``` java
class Solution {
  	// 线性增长的额外空间
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        ListNode head = null, tail = null;
        int add = 0;
        while (l1 != null || l2 != null || add != 0) {
            int sum = add;
            sum += l1 != null ? l1.val : 0;
            sum += l2 != null ? l2.val : 0;
            if (head == null) {
                head = tail = new ListNode(sum % 10);
            } else {
                tail.next = new ListNode(sum % 10);
                tail = tail.next;
            }

            if (l1 != null) l1 = l1.next;
            if (l2 != null) l2 = l2.next;

            add = sum / 10;
        }

        return head;
    }
}
```

```java
class Solution {
  	// 常数的额外空间
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        ListNode dummy = new ListNode();
        ListNode pre = dummy;
        dummy.next = l1;

        // 逐位相加
        int flag = 0;
        while (l1 != null && l2 != null) {
            int temp = l1.val + l2.val + flag;
            l1.val = temp % 10;
            flag = temp / 10;
            l1 = l1.next;
            pre = pre.next;
            l2 = l2.next;
        }
        // 高位继续相加
        ListNode prepre = pre;
        pre.next = (l1 != null ? l1 : l2);
        pre = pre.next;
        while (pre != null) {
            int temp = pre.val + flag;
            pre.val = temp % 10;
            flag = temp / 10;
            pre = pre.next;
            prepre = prepre.next;
        }
        // 最高位判断
        if (flag != 0) prepre.next = new ListNode(flag, null);

        return dummy.next;
    }
}
```



#### [8. 字符串转换整数 (atoi)](https://leetcode.cn/problems/string-to-integer-atoi/)

```` java
class Solution {
    public int myAtoi(String ss) {
        char[] s = ss.toCharArray();
        int left = 0, right = 0, flag = 0, isPos = 1;
        double res = 0;

        for (int i = 0; i < s.length; i++) {
            boolean isNum = (s[i] >= '0' && s[i] <= '9');
            if (!isNum && flag == 0) {
                if (s[i] != ' ' && s[i] != '+' && s[i] != '-') return 0;
                if (i == s.length - 1) return 0;
                if (s[i] == ' ') continue;
                if (!(s[i + 1] >= '0' && s[i + 1] <= '9')) return 0;
            }
            if (!isNum && flag == 1) break;
            if (isNum) {
                if (flag == 0) {
                    if (i > 0 && s[i - 1] == '-') isPos = 0;
                    left = i;
                    flag = 1;
                }
                right = i;
            }
        }

        if (flag == 0) return 0;

        for (int i = left; i <= right; i++) {
            res = res * 10;
            res += s[i] - '0';
        }
        res *= isPos == 1 ? 1 : -1;

        if (res > ((1 << 31) - 1)) res = ((1 << 31) - 1);
        if (res < (1 << 31) * -1) res = (1 << 31) * -1;

        return (int)res ;
    }
}
````



#### [93. 复原 IP 地址](https://leetcode.cn/problems/restore-ip-addresses/)

```` java
class Solution {
    List<String> res = new ArrayList<>();
    char[] path = new char[100];

    public List<String> restoreIpAddresses(String s) {
        char[] ss = s.toCharArray();
        backtracking(ss, 0, 0, -1);
        return res;
    }

    public void backtracking(char[] s, int start, int pointNums, int pathEnd) {
        if (pointNums == 3) {
            if (!isValid(s, start, s.length - 1)) return;
            for (int i = start; i < s.length; i++) path[++pathEnd] = s[i];
            res.add(new String(path, 0, pathEnd + 1));
        }

        int temp = pathEnd;
        for (int i = start; i <= start + 2 && i < s.length; i++) {
            if (!isValid(s, start, i)) continue;
            for (int j = start; j <= i; j++) path[++temp] = s[j];
            path[++temp] = '.';
            backtracking(s, i + 1, pointNums + 1, temp);
            temp = pathEnd;
        }
    }

    public boolean isValid(char[] s, int start, int end) {
        if (start > end || end - start > 2 || start > s.length - 1) return false;
        if (s[start] == '0' && start != end) return false;
        
        int sum = 0;
        for (int i = start; i <= end; i++) {
            sum *= 10;
            sum += s[i] - '0';
        }

        if (sum > 255) return false;
        return true;
    }
}
````



#### [239. 滑动窗口最大值](https://leetcode.cn/problems/sliding-window-maximum/)

``` java
class Solution {

    class myDeque {
        Deque<Integer> dq = new LinkedList<>();

        public myDeque() {

        }

        public void push(int x) {
            while (!dq.isEmpty() && x > dq.peekLast()) dq.pollLast();
            dq.offerLast(x);
        }

        public void pop(int x) {
            if (!dq.isEmpty() && dq.peekFirst() == x) dq.pollFirst();
        }

        public int peek() {
            if (!dq.isEmpty()) return dq.peekFirst();
            return -1;
        }
    }

    public int[] maxSlidingWindow(int[] nums, int k) {
        int[] res = new int[nums.length - k + 1];
        int index = 0;
        myDeque dq = new myDeque();
        for (int i = 0; i < k; i++) dq.push(nums[i]);
        res[index++] = dq.peek();

        for (int i = k; i < nums.length; i++) {
            dq.pop(nums[i - k]);
            dq.push(nums[i]);
            res[index++] = dq.peek();
        }

        return res;
    }
}
```



#### [41. 缺失的第一个正数](https://leetcode.cn/problems/first-missing-positive/)

``` java
class Solution {
    public int firstMissingPositive(int[] nums) { // 数组长度为n，如果这个数组中的所有1-n中间的数字按照对应的下标排序，那么从前往后遍历，第一个不符合对应关系的数字就是缺少的最小正数
        int n = nums.length;
        for (int i = 0; i < nums.length; i++) {
            while (nums[i] > 0 && nums[i] <= n && nums[nums[i] - 1] != nums[i]) { // 如果一个数不在正确的位置上，就给他交换到正确的位置
                int temp = nums[i];
                nums[i] = nums[nums[i] - 1];
                nums[temp - 1] = temp; 
            }
        }

        for (int i = 0; i < n; i++) {
            if (nums[i] != i + 1) return i + 1;
        }

        return n + 1;
    }
}
```



#### [剑指 Offer 22. 链表中倒数第k个节点](https://leetcode.cn/problems/lian-biao-zhong-dao-shu-di-kge-jie-dian-lcof/)

```` java
class Solution {
    public ListNode getKthFromEnd(ListNode head, int k) {
        ListNode dummy = new ListNode();
        dummy.next = head;
        ListNode fast = dummy, slow = dummy;
        while (k + 1 > 0) {
            k--;
            fast = fast.next;
        }

        while (fast != null) {
            fast = fast.next;
            slow = slow.next;
        }

        return slow.next;
    }
}
````



#### [76. 最小覆盖子串](https://leetcode.cn/problems/minimum-window-substring/)

滑动窗口

``` java
class Solution {
    public String minWindow(String ss, String tt) {
        char[] s = ss.toCharArray(), t = tt.toCharArray();
        int[] map = new int[100];
        int[] mark = new int[100];

        for (int i = 0; i < t.length; i++) {
            map[t[i] - 'A']++;
            mark[t[i] - 'A'] = 1;
        }

        int left = 0, right = 0, min = Integer.MAX_VALUE;
        int resLeft = -1, resRight = -1;
        while (right < s.length) {
            if (mark[s[right] - 'A'] == 1) map[s[right] - 'A']--;
            
            while (check(map) && left <= right) {
                if (right - left + 1 < min) {
                    resLeft = left;
                    resRight = right;
                    min = right - left + 1;
                }
                if (mark[s[left] - 'A'] == 1) map[s[left] - 'A']++;
                left++;
            }
            right++;
        }

        return resLeft == -1 ? "" : new String(s, resLeft, min);
    }

    public boolean check(int[] map) {
        for (int i = 0; i < map.length; i++) {
            if (map[i] > 0) return false;
        }
        return true;
    }
}
```



#### [105. 从前序与中序遍历序列构造二叉树](https://leetcode.cn/problems/construct-binary-tree-from-preorder-and-inorder-traversal/)

``` java
class Solution {
    public TreeNode buildTree(int[] preorder, int[] inorder) {
        return getTree(preorder, inorder, 0, preorder.length - 1, 0, inorder.length - 1);
    }

    // 表示从preorder和inorder的各自区域构建树
    public TreeNode getTree(int[] preorder, int[] inorder, int preStart, int preEnd, int inStart, int inEnd) {
        if (preEnd < preStart) return null;

        TreeNode node = new TreeNode(preorder[preStart]);
        // 找到前序的根节点在后序中的位置
        int index = 0;
        for (int i = inStart; i <= inEnd; i++) {
            if (inorder[i] == preorder[preStart]) {
                index = i;
                break;
            }
        }
        node.left = getTree(preorder, inorder, preStart + 1, preStart + index - inStart, inStart, index - 1);
        node.right = getTree(preorder, inorder, preStart + index - inStart + 1, preEnd, index + 1, inEnd);

        return node;
    }
}
```



#### [322. 零钱兑换](https://leetcode.cn/problems/coin-change/)

``` java
class Solution {
    public int coinChange(int[] coins, int amount) { // 完全背包，组合数
        // 从一堆数里选几个数，并满足一个要求，再求一个最优解
        int[] dp = new int[amount + 1];
        for (int i = 0; i <= amount; i++) dp[i] = Integer.MAX_VALUE;
        dp[0] = 0;

        for (int i = 1; i <= coins.length; i++) {
            for (int j = coins[i - 1]; j <= amount; j++) {
                if (dp[j - coins[i - 1]] == Integer.MAX_VALUE) continue;
                dp[j] = Math.min(dp[j], dp[j - coins[i - 1]] + 1);
            }
        }

        return dp[amount] == Integer.MAX_VALUE ? -1 : dp[amount];
    }
}
```



#### [165. 比较版本号](https://leetcode.cn/problems/compare-version-numbers/)

``` java
class Solution {
    public int compareVersion(String version1, String version2) {
        char[] v1 = version1.toCharArray(), v2 = version2.toCharArray();
        int left1 = 0, right1 = 0, left2 = 0, right2 = 0;
        while (right1 < v1.length && right2 < v2.length) {
            while (right1 < v1.length && v1[right1] != '.') right1++;
            while (right2 < v2.length && v2[right2] != '.') right2++;
            int res = compare(v1, left1, right1 - 1, v2, left2, right2 - 1);
            if (res == 0) {
                left1 = right1 + 1;
                left2 = right2 + 1;
                right1++;
                right2++;
            } else return res; 
        }

        if (right2 < v2.length && right1 >= v1.length) {
            while (right2 < v2.length) {
                while (right2 < v2.length && v2[right2] != '.') right2++;
                if (getSum(v2, left2, right2 - 1) != 0) {
                    // System.out.println(new String(v2, left2, right2 - left2));
                    return -1;
                }
                left2 = right2 + 1;
                right2++;
            }
        }
        if (right2 >= v2.length && right1 < v1.length) {
            while (right1 < v1.length) {
                while (right1 < v1.length && v1[right1] != '.') right1++;
                if (getSum(v1, left1, right1 - 1) != 0) return 1;
                left1 = right1 + 1;
                right1++;
            }
        }
        return 0;
    }

    public int getSum(char[] v, int l, int r) {
        int res = 0;
        for (int i = l; i <= r; i++) {
            res *= 10;
            res += v[i] - '0';
        }
        return res;
    }

    public int compare(char[] v1, int l1, int r1, char[] v2, int l2, int r2) {
        int res1 = getSum(v1, l1, r1);
        int res2 = getSum(v2, l2, r2);

        if (res1 > res2) return 1;
        else if (res1 < res2) return -1;
        return 0;
    }
}
```



#### [78. 子集](https://leetcode.cn/problems/subsets/)

``` java
class Solution {
    // 找出所有子集，是组合问题的一种，但是是找整个搜索树的所有节点
    // 组合问题，和切割问题，是找搜索树的叶子节点
    // 这两种搜索树其实都是根据一种思想得来：怎么找所有的组合，在单纯的组合问题中，k个位置的放置方法抽象出来是一颗搜索树；切割问题中，分割成k段的分割方法抽象出来也是一颗搜索树
    
    List<List<Integer>> res = new ArrayList<>();
    List<Integer> path = new ArrayList<>();

    public void backtrack(int[] nums, int startIndex) {
        res.add(new ArrayList<>(path));

        for (int i = startIndex; i < nums.length; i++) {
            path.add(nums[i]);
            backtrack(nums, i + 1);
            path.remove(path.size() - 1);
        }
    }
        
    public List<List<Integer>> subsets(int[] nums) {
        backtrack(nums, 0);
        return res;
    }
}
```



#### [43. 字符串相乘](https://leetcode.cn/problems/multiply-strings/)

``` java
class Solution {
    public String multiply(String num1, String num2) {
        if (num2.equals("0") || num1.equals("0")) return "0";
        char[] n1 = num1.toCharArray(), n2 = num2.toCharArray();
        int[] res = new int[500];
        int a = 0, b = 0;
        int n = n1.length, m = n2.length;
        for (int i = n - 1; i >= 0; i--) {
            a = n1[i] - '0';
            for (int j = m - 1; j >= 0; j--) {
                b = n2[j] - '0';
                res[i + j + 1] += a * b;
            }
        }

        for (int i = n + m - 1; i > 0; i--) {
            res[i - 1] += res[i] / 10;
            res[i] %= 10;
        }
        int index = res[0] == 0 ? 1 : 0;
        StringBuilder sb = new StringBuilder();
        // while (index < n + m - 1 && res[index] == 0) index++;
        for (int i = index; i < n + m; i++) sb.append(res[i]);

        return sb.toString();
    }
}
```



#### [200. 岛屿数量](https://leetcode.cn/problems/number-of-islands/)

深度优先搜索

``` java
class Solution {
    public int numIslands(char[][] grid) {
        int h = grid.length, w = grid[0].length;
        int res = 0;
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                if (grid[i][j] == '1') {
                    res++;
                    dfs(grid, i, j, h, w);            
                }
            }
        }

        return res;
    }

    public void dfs(char[][] grid, int i, int j, int h, int w) {
        if (i < 0 || i >= h || j < 0 || j >= w || grid[i][j] == '0') return;

        grid[i][j] = '0';
        dfs(grid, i - 1, j, h, w);
        dfs(grid, i + 1, j, h, w);
        dfs(grid, i, j - 1, h, w);
        dfs(grid, i, j + 1, h, w);
    }
}
```

 

#### [179. 最大数](https://leetcode.cn/problems/largest-number/)

``` java
class Solution {
    public String largestNumber(int[] num) {
        Integer[] nums = new Integer[num.length]; // 转换成包装类型
        for (int i = 0; i < num.length; i++) nums[i] = num[i];
        Arrays.sort(nums, (a, b) -> {
            int res = compare(a, b);
            return res;
        });
        if (nums[0] == 0) return "0";
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < nums.length; i++) sb.append(nums[i]);

        return sb.toString();
    }

    public int compare(int a, int b) {
        long apix = 10, bpix = 10;
        while (apix <= b) {
            apix *= 10;
        }
        while (bpix <= a) {
            bpix *= 10;
        }

        if (a * apix + b - b * bpix - a > 0) return -1; // a在前面更大
        return 1;
    }


}
```



#### [104. 二叉树的最大深度](https://leetcode.cn/problems/maximum-depth-of-binary-tree/)

``` java
class Solution {
    public int maxDepth(TreeNode root) {
        if (root == null) return 0;

        return Math.max(maxDepth(root.left), maxDepth(root.right)) + 1;
    }
}
```



#### [155. 最小栈](https://leetcode.cn/problems/min-stack/)

``` java
class MinStack {
    Deque<Integer> st1 = new LinkedList<>(); // 存数据
    Deque<Integer> st2 = new LinkedList<>(); // 维护最小值

    public MinStack() {

    }
    
    public void push(int val) {
        st1.push(val);
        if (st2.isEmpty()) st2.push(val);
        else {
            if (val < st2.peek()) st2.push(val);
            else st2.push(st2.peek());
        }
    }
    
    public void pop() {
        st1.pop();
        st2.pop();
    }
    
    public int top() {
        return st1.peek();
    }
    
    public int getMin() {
        return st2.peek();
    }
}

/**
 * Your MinStack object will be instantiated and called as such:
 * MinStack obj = new MinStack();
 * obj.push(val);
 * obj.pop();
 * int param_3 = obj.top();
 * int param_4 = obj.getMin();
 */
```



#### [110. 平衡二叉树](https://leetcode.cn/problems/balanced-binary-tree/)

``` java
class Solution {
    public boolean isBalanced(TreeNode root) {
        int res = checkBalanced(root);
        return res != -1;
    }

    public int checkBalanced(TreeNode root) {
        if (root == null) return 0;

        int left = checkBalanced(root.left);
        if (left == -1) return -1;
        int right = checkBalanced(root.right);
        if (right == -1) return -1;

        if (left > right + 1 || right > left + 1) return -1;
        
        return left > right ? left + 1 : right + 1;
    }
}
```



#### [32. 最长有效括号](https://leetcode.cn/problems/longest-valid-parentheses/)

```` java
class Solution {
    public int longestValidParentheses(String ss) {
        char[] s = ss.toCharArray();
        int max = 0, left = 0, right = 0;
        for (int i = 0; i < s.length; i++) {
            if (s[i] == '(') left++;
            else right++;
            if (left < right) {
                left = 0;
                right = 0;
            } else if (left == right) {
                max = Math.max(max, left * 2);
            }
        }

        left = 0;
        right = 0;
        for (int i = s.length - 1; i >= 0; i--) {
            if (s[i] == ')') right++;
            else left++;
            if (right < left) {
                left = 0;
                right = 0;
            } else if (left == right) {
                max = Math.max(max, left * 2);
            }
        }

        return max;
    }
}
````

``` java
class Solution {
    public int longestValidParentheses(String ss) {
        char[] s = ss.toCharArray();
        int[] dp = new int[s.length];
        int max = 0;
        for (int i = 1; i < s.length; i++) {
            if (s[i] == ')') {
                if (s[i - 1] == '(') dp[i] = i > 2 ? dp[i - 2] + 2 : 2;
                else if (i - dp[i - 1] > 0 && s[i - dp[i - 1] - 1] == '(') {
                    dp[i] = dp[i - 1] + 2 + (i - dp[i - 1] - 2 >= 0 ? dp[i - dp[i - 1] - 2] : 0);
                }
                max = Math.max(max, dp[i]);
            }
        }

        return max;
    }
}
```



#### [101. 对称二叉树](https://leetcode.cn/problems/symmetric-tree/)

``` java
// class Solution {
//     public boolean isSymmetric(TreeNode root) {
//         if (root == null) return true;
//         return isTrue(root.left, root.right);
//     }

//     public boolean isTrue(TreeNode root1, TreeNode root2) {
//         if (root1 == null && root2 == null) return true;
//         if (!(root1 != null && root2 != null)) return false;
//         if (root1.val != root2.val) return false;
//         return isTrue(root1.left, root2.right) && isTrue(root1.right, root2.left);
//     }
// }

class Solution {
    public boolean isSymmetric(TreeNode root) {
        if (root == null) return true;
        
        Queue<TreeNode> q = new LinkedList<>();
        q.offer(root.left);
        q.offer(root.right);
        while (!q.isEmpty()) {
            TreeNode node1 = q.poll();
            TreeNode node2 = q.poll();
            if (node1 == null && node2 == null) continue;
            if (!(node1 != null && node2 != null)) return false;
            if (node1.val != node2.val) return false;
            
            q.offer(node1.left);
            q.offer(node2.right);

            q.offer(node1.right);
            q.offer(node2.left);
        }

        return true;
    }
}
```



#### [129. 求根节点到叶节点数字之和](https://leetcode.cn/problems/sum-root-to-leaf-numbers/)

``` java
class Solution {
    int res = 0;
    public int sumNumbers(TreeNode root) {
        backtrack(root, 0);
        return res;
    }

    public void backtrack(TreeNode root, int x) {
        if (root.left == null && root.right == null) {
            res += x * 10 + root.val;
            return;
        }

        int temp = x * 10 + root.val;
        if (root.left != null) backtrack(root.left, temp);
        if (root.right != null) backtrack(root.right, temp);
        return;
    }
}
```



#### [543. 二叉树的直径](https://leetcode.cn/problems/diameter-of-binary-tree/)

``` java
class Solution {
    int res = 0;
    public int diameterOfBinaryTree(TreeNode root) {
        backtrack(root);
        return res - 1;
    }

    public int backtrack(TreeNode root) {
        if (root == null) return 0;

        int left = backtrack(root.left);
        int right = backtrack(root.right);

        res = Math.max(res, left + right + 1);
        return Math.max(left, right) + 1;
    }
}
```



#### [113. 路径总和 II](https://leetcode.cn/problems/path-sum-ii/)

```` java
class Solution {
    List<List<Integer>> res = new ArrayList<>();
    List<Integer> path = new ArrayList<>();

    public List<List<Integer>> pathSum(TreeNode root, int targetSum) {
        if (root == null) return res;
        path.add(root.val);
        backtrack(root, root.val, targetSum); // 当前要从root的子节点开始选择
        return res;
    }

    public void backtrack(TreeNode root, int sum, int target) {
        if (root.left == null && root.right == null) {
            if (sum != target) return;
            res.add(new ArrayList<>(path));
            return;
        }

        if (root.left != null) { // 选择左子节点
            path.add(root.left.val);
            backtrack(root.left, sum + root.left.val, target);
            path.remove(path.size() - 1);
        }
        if (root.right != null) { // 选择右子节点
            path.add(root.right.val);
            backtrack(root.right, sum + root.right.val, target);
            path.remove(path.size() - 1);
        }
    }
}
````



#### [98. 验证二叉搜索树](https://leetcode.cn/problems/validate-binary-search-tree/)

``` java
class Solution {
    TreeNode pre = null;

    public boolean isValidBST(TreeNode root) { // 二叉搜索树的元素排列顺序就是，中序遍历的顺序，所以只要按照中序遍历，并记录pre节点，进行比价即可
        if (root == null) return true;

        boolean left = isValidBST(root.left);
        if (pre != null && pre.val >= root.val) return false;
        pre = root;

        boolean right = isValidBST(root.right);

        return left && right;
    }
}
```



#### [64. 最小路径和](https://leetcode.cn/problems/minimum-path-sum/)

``` java
class Solution {
    public int minPathSum(int[][] grid) {
        int[][] dp = new int[grid.length][grid[0].length];
        
        dp[0][0] = grid[0][0];
        for (int i = 1; i < grid.length; i++) dp[i][0] = dp[i - 1][0] + grid[i][0];
        for (int i = 1; i < grid[0].length; i++) dp[0][i] = dp[0][i - 1] + grid[0][i];

        for (int i = 1; i < grid.length; i++) {
            for (int j = 1; j < grid[0].length; j++) {
                dp[i][j] = Math.min(dp[i - 1][j], dp[i][j - 1]) + grid[i][j];
            }
        }

        return dp[grid.length - 1][grid[0].length - 1];
    }
}
```



#### [48. 旋转图像](https://leetcode.cn/problems/rotate-image/)

``` java
class Solution {
    public void rotate(int[][] matrix) {
        int n = matrix.length;
        // 水平翻转
        for (int i = 0; i < n / 2; i++) {
            for (int j = 0; j < n; j++) {
                int temp = matrix[i][j];
                matrix[i][j] = matrix[n - 1 - i][j];
                matrix[n - 1 - i][j] = temp;
            }
        }
				
      	// 对角线翻转
        for (int i = 0; i < n; i++) {
            for (int j = 0; j <= i; j++) {
                int temp = matrix[i][j];
                matrix[i][j] = matrix[j][i];
                matrix[j][i] = temp;
            }
        }
    }
}
```



#### [470. 用 Rand7() 实现 Rand10()](https://leetcode.cn/problems/implement-rand10-using-rand7/)

拒绝采样

``` java
class Solution extends SolBase {
    public int rand10() {
        while (true) {
            int a = rand7();
            int b = rand7();
            int c = (a - 1) * 7 + b;
            if (c <= 40) return c % 10 + 1;

            b = c - 40;
            a = rand7();
            c = (b - 1) * 7 + a;
            if (c <= 60) return c % 10 + 1;

            b = c - 60;
            a = rand7();
            c = (b - 1) * 7 + a;
            if (c <= 20) return c % 10 + 1;
        }
    }
}
```



#### [112. 路径总和](https://leetcode.cn/problems/path-sum/)

```` java
class Solution {
    public boolean hasPathSum(TreeNode root, int targetSum) {
        if (root == null) return false;
        return trackback(root, targetSum, root.val); // 从当前节点的子节点开始搜索,因为需要判断叶子节点
    }

    public boolean trackback(TreeNode root, int targetSum, int sum) {
        if (root.left == null && root.right == null && sum == targetSum) return true; 

        if (root.left != null) {
            boolean left = trackback(root.left, targetSum, sum + root.left.val);
            if (left) return true;
        }

        if (root.right != null) {
            boolean right = trackback(root.right, targetSum, sum + root.right.val);
            if (right) return true;
        }

        return false;
    }
}
````



#### [24. 两两交换链表中的节点](https://leetcode.cn/problems/swap-nodes-in-pairs/)

```java
class Solution {
    // 将链表分为很多的2元祖，在2元祖内部进行链表反转，再进行合并
  	// 需要考虑的点：
  	// 1. 虚拟头指针
  	// 2. 翻转一共需要几个指针和步骤
  	// 3. 循环退出的条件是什么
    public ListNode swapPairs(ListNode head) {
        if (head == null || head.next == null) return head;

        ListNode danny = new ListNode();
        danny.next = head;

        ListNode pre = danny, cur = head, next = null; 
				// cur指向下一轮翻转的第一个节点，下一次反转开始判断当前轮是否满足翻转的需要
        while (!(cur == null || cur.next == null)) {
          	// 翻转和拼接
            next = cur.next;
            cur.next = next.next;
            next.next = cur;
            pre.next = next;
						
            pre = cur;
            cur = cur.next;
        }

        return danny.next;
    }
}
```



#### [141. 环形链表](https://leetcode.cn/problems/linked-list-cycle/)

```java
public class Solution {
    // 快慢指针
    public boolean hasCycle(ListNode head) {
        ListNode fast = head, slow = head;

        while (!(fast == null || fast.next == null)) {
            slow = slow.next;
            fast = fast.next.next;
            if (fast == slow) return true;
        }

        return false;
    }
}
```



#### [92. 反转链表 II](https://leetcode.cn/problems/reverse-linked-list-ii/)

```java
class Solution {
    public ListNode reverseBetween(ListNode head, int left, int right) {
        ListNode dummy = new ListNode();
        dummy.next = head;
        ListNode start = dummy, end = dummy;

        while (left - 1 > 0) {
            start = start.next;
            left--;
        }
        while (right > 0) {
            end = end.next;
            right--;
        }
        ListNode next = end.next;
        reverse(start.next, end);
        start.next.next = next;
        start.next = end;

        return dummy.next;
    }

    // 翻转链表
    public void reverse(ListNode start, ListNode end) {
        ListNode pre = null, cur = start, next = null;

        while (cur != end) {
            next = cur.next;
            cur.next = pre;
            pre = cur;
            cur = next;
        }
        cur.next = pre;
    }
}
```



#### [234. 回文链表](https://leetcode.cn/problems/palindrome-linked-list/)

```java
class Solution {
    public boolean isPalindrome(ListNode head) {
        if (head.next == null) return true;

        int length = 0;
        ListNode cur = head;
        while (cur != null) {
            length++;
            cur = cur.next;
        }
        int half = length / 2 + length % 2;
        cur = head;
        while (half > 0) {
            half--;
            cur = cur.next;
        }

        ListNode newHead = reverse(cur);

        while (newHead != null) {
            if (newHead.val != head.val) return false;
            newHead = newHead.next;
            head = head.next;
        }
        return true;
    }

    public ListNode reverse(ListNode head) {
        ListNode pre = null, cur = head, next = null;

        while (cur.next != null) {
            next = cur.next;
            cur.next = pre;
            pre = cur;
            cur = next;
        }
        cur.next = pre;
        return cur;
    }
}
```



#### [83. 删除排序链表中的重复元素](https://leetcode.cn/problems/remove-duplicates-from-sorted-list/)

```java
class Solution {
    public ListNode deleteDuplicates(ListNode head) {
        if (head == null || head.next == null) return head;

        ListNode cur = head;

        // cur指向当前需要判断的节点,由于和后续节点相关，因为退出条件是两个节点都不为空
        while (cur != null && cur.next != null) {
            if (cur.val == cur.next.val) {
                int target = cur.val;
                ListNode next = cur.next;
                while (next != null && next.val == target) next = next.next;
                cur.next = next;
            }
            cur = cur.next;
        }

        return head;
    }
}
```



#### [138. 复制带随机指针的链表](https://leetcode.cn/problems/copy-list-with-random-pointer/)

```java
class Solution {
    public Node copyRandomList(Node head) {
        if (head == null) return null;
        // 先遍历一遍，用哈希表记录指针地址和序号，并创建新链表，并记录新链表的指针地址和序号的哈希表
        // 再遍历一遍，根据对应关系，填充新链表的randeom

        // copy 的链表需要知道原先链表中对应位置的节点指向的是自己的第几个节点
        HashMap<Node, Integer> map1 = new HashMap<Node, Integer>();
        HashMap<Integer, Node> map2 = new HashMap<Integer, Node>();
        Node dummy = new Node(0);
        Node l1 = head, l2 = dummy;
        int index = 0;

        while (l1 != null) {
            map1.put(l1, index);
            l2.next = new Node(l1.val);
            map2.put(index, l2.next);
            l2 = l2.next;
            l1 = l1.next;
            index++;
        }
        l2.next = null;

        l1 = head;
        l2 = dummy.next;
        while (l1 != null) {
            if (l1.random == null) {
                l2.random = null;
            } else {
                l2.random = map2.get(map1.get(l1.random));
            }
            l1 = l1.next;
            l2 = l2.next;
        }j
        return dummy.next;
    }
}
```



#### [61. 旋转链表](https://leetcode.cn/problems/rotate-list/)

```java
class Solution {
    public ListNode rotateRight(ListNode head, int k) {
        if (head == null || head.next == null) return head;
        ListNode cur = head;
        int length = 0;
        
        while (cur != null) {
            length++;
            cur = cur.next;
        }
        
        int realK = k % length;
        if (realK == 0) return head;
        int index = length - realK;

        cur = head;
        while (index - 1 > 0) {
            index--;
            cur = cur.next;
        }

        ListNode newHead = cur.next;
        cur.next = null;
        cur = newHead;
        while (cur.next != null) cur = cur.next;
        cur.next = head;

        return newHead;
    }
}
```



#### [106. 从中序与后序遍历序列构造二叉树](https://leetcode.cn/problems/construct-binary-tree-from-inorder-and-postorder-traversal/)

```java
class Solution {
    public TreeNode buildTree(int[] inorder, int[] postorder) {
        return getTree(inorder, postorder, 0, inorder.length - 1, 0, postorder.length - 1);
    }

    public TreeNode getTree(int[] inorder, int[] postorder, int inStart, int inEnd, int postStart, int postEnd) {
        if (inStart > inEnd) return null;

        TreeNode node = new TreeNode(postorder[postEnd]);
        int index = 0;
        for (int i = inStart; i <= inEnd; i++) {
            if (inorder[i] == postorder[postEnd]) {
                index = i;
                break;
            }
        }

        node.left = getTree(inorder, postorder, inStart, index - 1, postStart, postStart + index - inStart - 1);
        node.right = getTree(inorder, postorder, index + 1, inEnd, postStart + index - inStart, postEnd - 1);

        return node;
    }
}
```



#### [226. 翻转二叉树](https://leetcode.cn/problems/invert-binary-tree/)

```java
class Solution {
    // 先递归翻转，最后把根节点翻转，然后返回翻转好的节点
    public TreeNode invertTree(TreeNode root) {
        if (root == null) return null;
        TreeNode right = invertTree(root.left);
        TreeNode left = invertTree(root.right);
        root.left = left;
        root.right = right;
        return root;
    }
}
```

