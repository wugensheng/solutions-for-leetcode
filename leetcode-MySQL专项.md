## 第1天——选择

#### [595. 大的国家](https://leetcode.cn/problems/big-countries/)

``` sql
# Write your MySQL query statement below
select name, population, area 
from World 
where area >= 3000000 or population >= 25000000;
```

#### [1757. 可回收且低脂的产品](https://leetcode.cn/problems/recyclable-and-low-fat-products/)

```` sql
# Write your MySQL query statement below
select product_id
from Products
where low_fats = 'Y' and recyclable = 'Y';
````



#### [584. 寻找用户推荐人](https://leetcode.cn/problems/find-customer-referee/)

```` sql
# Write your MySQL query statement below
select name
from customer
where referee_id != 2 or referee_id is null; # msyql三值逻辑，任何与null的比较都会变成unknown, 包括null自己；所以需要用is null
````



#### [183. 从不订购的客户](https://leetcode.cn/problems/customers-who-never-order/)

``` sql
# Write your MySQL query statement below
select Name as Customers
from Customers
where id not in ( # 子查新
    select CustomerId
    from Orders
);
```



## 第2天——排序修改

#### [1873. 计算特殊奖金](https://leetcode.cn/problems/calculate-special-bonus/)

``` sql
# Write your MySQL query statement below
select 
    employee_id,
    case 
        when mod(employee_id, 2) != 0 and left(name, 1) != 'M' then salary
        # when mod(employee_id, 2) = 0 or left(name, 1) != 'M' then 0
        else 0
    end as bonus
from Employees
order by employee_id
```



#### [627. 变更性别](https://leetcode.cn/problems/swap-salary/)

``` sql
# Write your MySQL query statement below
update Salary
set
    sex = case
            when sex = 'm' then 'f'
            else 'm'
    end;

```



#### [196. 删除重复的电子邮箱](https://leetcode.cn/problems/delete-duplicate-emails/)

``` sql
# Please write a DELETE statement and DO NOT write a SELECT statement.
# Write your MySQL query statement below
delete p1 from Person p1, Person p2
where p1.Email = p2.Email and p1.Id > p2.Id;
```





