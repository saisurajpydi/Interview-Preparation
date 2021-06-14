"""
Task 2: Stella and colorful tree

Problem Statement: 
       While returning from Byteland, Stella got one tree with N nodes from her friend over there. All nodes in this tree are colorless and Stella decided to fill colors to make it colorful. Stella wants it to look beautiful and decided to color it in such a way that any 2 nodes u and v with the shortest distance between u and v <=2 can not be of the same color. She is wondering how many different colors she needs if she fills optimally.

 Input Format:
         The first line contains a single integer n ( 3 <=n<100) – the number of nodes in the tree. Each of the next(n-1) lines contains two integers x and y(1<=x, y<=n) – the indices of two nodes directly connected by an edge. It is guaranteed that any node is reachable from any other using the edges.

Output Format:
          In the first line print single integer k – the minimum number of colors Stell has to use.

Sample Input 1:
           3
          2 3
          1 3
Sample Output 1:
          3
 Explanation Output 1:
           Tree is like
           1 -> 3 ->2
We can color as follows
         1: Color a
         3: Color b
         2 : Color c
         Total 3 colors
Sample Input 2:
           5
          2 1
          3 2
          4 3
          5 4
Sample Output 2
           3
Explanation Output2: 
          Tree is like:
          1 -> 2 ->3 -> 4 -> 5
         We can color as follows
         1: Color a
         2: Color b
         3 : Color c
         4: Color a
         5 : Color b
         Total 3 colors.

"""
n = int(input())
dic = dict()
for i in range(1,n):
    x, y = map(int, input().split()) 
    dic[x] = y 
    

