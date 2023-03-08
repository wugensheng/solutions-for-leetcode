

####  [3. 无重复字符的最长子串](https://leetcode.cn/problems/longest-substring-without-repeating-characters/)

``` java
class Solution {
    public int lengthOfLongestSubstring(String s) {
        char[] s = ss.toCharArray();
        if (s.length <= 1) return s.length;
        int[] mark = new int[256];
        int res = 0;

        int left = 0, right = 0;
        while (right < s.length){ // 滑动窗口 维护一个不重复序列，不重复序列用哈希表管理
            if (mark[s[right]] == 0) {
                mark[s[right]] = 1;
                right++;
                if (right - left > res) res = right - left; 
                continue;
            }
            while(s[left] != s[right]) {
                mark[s[left]] = 0;
                left++;
            }
            left++;
            right++;
        }
        return res;
    }
}
```



#### [25. K 个一组翻转链表](https://leetcode.cn/problems/reverse-nodes-in-k-group/)

``` java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode() {}
 *     ListNode(int val) { this.val = val; }
 *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
 * }
 */
class Solution {
    public ListNode reverseKGroup(ListNode head, int k) {
        ListNode dummy = new ListNode();
        dummy.next = head;
        ListNode start = head, end = null, pre = dummy;
        while (start != null) {
            end = pre;
            for (int i = 0; i < k; i++) {
                end = end.next;
                if (end == null) {
                    return dummy.next;
                }
            }
            // 开始翻转
            ListNode next = end.next;
            reverse(start, end);
            start.next = next; // 头放尾部
            pre.next = end; // 尾部接pre后面
            pre = start;
            start = pre.next;
        }
        return dummy.next;
    }

    public void reverse(ListNode start, ListNode end) {
        ListNode cur = start, pre = null, temp = null;
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



#### [146. LRU 缓存](https://leetcode.cn/problems/lru-cache/)

哈希表+双向链表

``` java
class LRUCache {
    // 双向链表数据结构
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
    private DLinkedNode head, tail; // 虚拟头节点和虚拟尾节点

    public LRUCache(int capacity) {
        // 初始化
        this.capacity = capacity;
        this.size = 0;
        // 使用虚拟头节点和虚拟尾节点
        head = new DLinkedNode();
        tail = new DLinkedNode();
        head.next = tail;
        tail.pre = head;
    }
    
    public int get(int key) {
        DLinkedNode node = cache.get(key);
        if (node == null) return -1;
        // 移动到双向链表头部
        moveToHead(node);
        return node.value;
    }
    
    public void put(int key, int value) {
        DLinkedNode node = cache.get(key);
        if (node == null) { // 节点不存在，创建新节点
            DLinkedNode newNode = new DLinkedNode(key, value);
            cache.put(key, newNode);
            size++;
            addToHead(newNode);

            if (size > capacity) {
                DLinkedNode tail = removeTail();
                cache.remove(tail.key);
                size--;
            }

        } else { // 节点存在，更新节点
            node.value = value;
            moveToHead(node);
        }
    }

    public void moveToHead(DLinkedNode node) {
        removeNode(node);
        addToHead(node);
    }

    public DLinkedNode removeTail() {
        DLinkedNode node = tail.pre;
        removeNode(node);
        return node; 
    }

    public void addToHead(DLinkedNode node) {
        node.pre = head;
        node.next = head.next;
        head.next = node;
        node.next.pre = node;
    }

    public void removeNode(DLinkedNode node) {
        node.pre.next = node.next;
        node.next.pre = node.pre;
    }
}

/**
 * Your LRUCache object will be instantiated and called as such:
 * LRUCache obj = new LRUCache(capacity);
 * int param_1 = obj.get(key);
 * obj.put(key,value);
 */
```



#### [215. 数组中的第K个最大元素](https://leetcode.cn/problems/kth-largest-element-in-an-array/)

``` java
class Solution {
    public int findKthLargest(int[] nums, int k) {
        return quickSelect(nums, 0, nums.length - 1, nums.length - k);
    }

    public int quickSelect(int[] nums, int l, int r, int index) {
        int pos = randomSelect(nums, l, r);
        if (pos == index) return nums[pos]; // 根据index进行加速
        else {
            return pos > index ? quickSelect(nums, l, pos - 1, index) : quickSelect(nums, pos + 1, r, index);
        }
    }

    public int randomSelect(int[] nums, int l, int r) {
        int i = new Random().nextInt(r - l + 1) + l;
        swap(nums, i, r);
        return partition(nums, l, r);
    }

    public int partition(int[] nums, int l, int r) {
        int flag = nums[r];
        int i = l - 1;
        for (int j = l; j <= r - 1; j++) {
            if (nums[j] <= flag) {
                i++;
                swap(nums, i, j);
            }
        }
        swap(nums, i + 1, r);
        return i + 1;
    }

    public void swap(int[] nums, int l, int r) {
        int temp = nums[l];
        nums[l] = nums[r];
        nums[r] = temp;
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
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode() {}
 *     ListNode(int val) { this.val = val; }
 *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
 * }
 */
class Solution {
    public ListNode mergeTwoLists(ListNode list1, ListNode list2) {
        ListNode dummy = new ListNode();
        ListNode head1 = list1, head2 = list2, cur = dummy;
        while (head1 != null && head2 != null) {
            if (head1.val < head2.val) {
                cur.next = head1;
                head1 = head1.next;
            } else {
                cur.next = head2;
                head2 = head2.next;
            }
            cur = cur.next;
        }
        while (head1 != null) {
            cur.next = head1;
            head1 = head1.next;
            cur = cur.next;
        }
        while (head2 != null) {
            cur.next = head2;
            head2 = head2.next;
            cur = cur.next;
        }

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
        List<List<Integer>> res = new ArrayList<>();
        if (root == null) return res;

        Queue<TreeNode> q = new LinkedList<>();
        q.offer(root);
        while (!q.isEmpty()) {
            int size = q.size();
            List<Integer> list = new ArrayList<>();
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



#### [33. 搜索旋转排序数组](https://leetcode.cn/problems/search-in-rotated-sorted-array/)

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
        boolean fromLeft = true; // 表示当前层的结果顺序
        q.offer(root);
        while (!q.isEmpty()) {
            int size = q.size();
            Deque<Integer> dq = new LinkedList<Integer>();
            for (int i = 0; i < size; i++) {
                TreeNode node = q.poll();
                if (fromLeft) { // 当前层节点从做到右放入dq中
                    dq.offerLast(node.val);
                } else {
                    dq.offerFirst(node.val);
                }
                if (node.left != null) q.offer(node.left);
                if (node.right != null) q.offer(node.right);
            }
            res.add(new LinkedList<>(dq));
            fromLeft = !fromLeft; // 取反
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
        ListNode curA = headA, curB = headB;
        int countA = 0, countB = 0;
        while (curA != null) {
            countA++;
            curA = curA.next;
        }
        while (curB != null) {
            countB++;
            curB = curB.next;
        }

        int des = countA > countB ? countA - countB : countB - countA;
        curA = headA;
        curB = headB;
        if (countA > countB) {
            while (des > 0) {
                curA = curA.next;
                des--;
            }
        } else {
            while (des > 0) {
                curB = curB.next;
                des--;
            }
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
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode() {}
 *     ListNode(int val) { this.val = val; }
 *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
 * }
 */
class Solution { // 分治思想优化顺序合并
    public ListNode mergeKLists(ListNode[] lists) {
        ListNode res = merge(lists, 0, lists.length - 1);
        return res;
    }
    
    public ListNode merge(ListNode[] lists, int l, int r) {
        if (l == r) return lists[l];
        if (l > r) return null;
        int mid = (l + r) >> 1;
        ListNode left = merge(lists, l, mid); // 先各自解决子问题
        ListNode right = merge(lists, mid + 1, r);
        return mergeTwoList(left, right); // 最后做总处理
    }

    public ListNode mergeTwoList(ListNode list1, ListNode list2) {
        ListNode dummy = new ListNode();
        ListNode cur1 = list1, cur2 = list2, cur = dummy;
        while (cur1 != null && cur2 != null) {
            if (cur1.val < cur2.val) {
                cur.next = cur1;
                cur1 = cur1.next;
            } else {
                cur.next = cur2;
                cur2 = cur2.next;
            }
            cur = cur.next;
        }
        cur.next = cur1 == null ? cur2 : cur1;
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
    public ListNode detectCycle(ListNode head) {
        ListNode fast = head, slow = head;

        while (fast != null && fast.next != null) {
            fast = fast.next.next;
            slow = slow.next;
            if (slow == fast) {
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
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode() {}
 *     ListNode(int val) { this.val = val; }
 *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
 * }
 */
class Solution {
    public void reorderList(ListNode head) {
        ListNode dummy = new ListNode();
        dummy.next = head;
        ListNode cur = dummy;
        int count = 0;
        while (cur.next != null) {
            cur = cur.next;
            count++;
        }

        if (count <= 2) return;

        int half = (count + 1) / 2;
        cur = dummy;
        while (half > 0) {
            cur = cur.next;
            half--;
        }

        ListNode left = head, right = cur.next;
        cur.next = null;
        cur = dummy;
        right = reverse(right);

        while (left != null && right != null) {
            cur.next = left;
            left = left.next;
            cur = cur.next;
            cur.next = right;
            right = right.next;
            cur = cur.next;
        }
        cur.next = left;
    }

    public ListNode reverse(ListNode head) {
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
    public ListNode removeNthFromEnd(ListNode head, int n) {
        ListNode dummy = new ListNode();
        dummy.next = head;

        ListNode fast = dummy, slow = dummy;
        while (n > 0) {
            fast = fast.next;
            n--;
        }

        while (fast.next != null) {
            fast = fast.next;
            slow = slow.next;
        }

        slow.next = slow.next.next;

        return dummy.next;
    }
}
```



#### [199. 二叉树的右视图](https://leetcode.cn/problems/binary-tree-right-side-view/)

``` java
class Solution {
    public List<Integer> rightSideView(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        if (root == null) return res;

        Queue<TreeNode> q = new LinkedList<>();
        q.offer(root);
        
        while (!q.isEmpty()) {
            int size = q.size();
            for (int i = 0; i < size; i++) {
                TreeNode node = q.poll();
                if (i ==  size - 1) {
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

