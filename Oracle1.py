"""
Task 1: Girls in Tech Hackathon

Problem Statement:
        The Girl in Tech Hackathon is code-a-thon where developers, designers, scientists, students, entrepreneurs, and educators gather to collaborate on projects including applications, software, hardware, data visualization, and platform solutions. The Participants are sitting around the table, and they are numbered from team1 to team n in the clockwise direction. This means that the Participants are numbered 1,2, ..,n-1,n and participants 1 and n are sitting next to each other. After the Hackathon duration, judges started reviewing the performance of each participant. However, some participants have not finished the project. Each participant needs ti extra minute to complete their project. Judges started reviewing the projects sequentially in the clock direction, starting with team x, and it takes them exactly 1 minute to review each project. This means team x gets no extra time to work, whereas team x +1 gets 1 min more to work and so on. Even if the project is not completed by the participant, but still the judges spend 1 min to understand the idea and rank the participant.

Input Format:
         The first line contains a single positive integer, n, denoting the number of participants in the hackathon. Given the values of t1,t2, t3,… tn extra time requires by each participant to complete the project. You have to guide judges in selecting the best team x to start reviewing the project so that number of participants able to complete their project is maximal.

Output Format:
Print x on a new line. If there are multiple such teams, 
select the smallest one.
Constraints:
1<=n<=9*10^5
0<=ti<=n
Sample Input:
 3
1 0 0
Sample Output:
 2
"""
"""
n = 8
    [6,4,2,5,1,4,8,0]
    [0,1,2,4,4,5,6,8]
"""
n = int(input())
times = list(map(int, input().split()))
val = min(times)
res = 0
for i in range(n):
    if(times[i] == val):
        res = i
        break
print(res+1)
