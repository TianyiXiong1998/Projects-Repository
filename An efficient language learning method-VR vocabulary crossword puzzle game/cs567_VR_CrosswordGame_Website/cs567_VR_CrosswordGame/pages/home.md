Group Member: Jieqiong Li, Zihui Li, Tianyi Xiong

# Introduction:
Whether it is child education or adult education,
there is no doubt that language learning is one of 
the most significant parts of education. Language education
is an important education direction that can expand its horizons. 
As we all know, if a person wants to learn a new language, 
memorizing words is the first step in learning. 
However, some people still use the traditional methods to 
learn vocabulary, such as repeating words one by one, 
which also makes the learning language efficiency lower and lower. 
In addition, as a discipline that requires independent learning, 
if the student does not have enough interest to learn a new language, 
they will have high anxiety and pressure when reciting the vocabulary.
 How to make students more efficient and happier learning language will 
 be the main direction to language education. 
 So we hope to help students learn English words more easily and 
 happily by creating a word game to learn. 
 And use the most influential technology, 
 Virtual Reality (VR) technology,  to further enhance students' 
 interest in learning. We have three purposes for this topic. 
 First, we want to break traditional learning methods and improve 
 the efficiency of students when reciting words. Second, 
 we hope that when people use this game, they can be immersed in 
 a more relaxed learning atmosphere, enjoy the fun of learning words. 
 Last but not least, we hope that through the analysis of experimental 
 results, we have learned whether VR technology has a good influence 
 on education.


# Motivation:

In 2019, Jing\[1\] explains why students are anxious when learning 
languages. After we combined Jing’s discovery with our own 
learning experience, we found that learning methods and 
learning interests are the two main reasons that affect 
language learning. The first is the
problem of learning methods. Students need to have strong 
self-learning ability and perfect learning methods in order to 
learn a second language well. If students only learn the language 
by using the traditional method of memorizing words, 
they often fail to obtain a good result. 
Secondly, students will gradually lose interest in completing the 
boring task of reciting words. Therefore, we hope to create a game 
that can improve students' learning interests and learning efficiency 
to help students learn a second language happily. In the article, 
"The Effect of Puzzle Game on Students' Vocabulary Achievement Forum, 
Juliana (2020) explains the availability of crossword puzzle games 
for English teaching \[2\]. And Juliana points out that the game helps 
students raise their interests and helps to improve students' 
efficiency of the second language. She proposed that after using 
the crossword game, the students' vocabulary skill has obviously 
improved. In addition, the competition and time stress in the game 
will also give students more active in the process of playing 
games \[3\]\[4\]. As the most popular 3D technology, VR technology 
is an optimal technology that attracts students and enhances 
students' interest in learning. 3D technology can help students 
improve their memory and implement an immersive 
learning environment \[5\]. If we can combine the crosswords 
and VR technologies, we will give students a new experience in 
reciting words. We also hope that by using this method, students 
will get the ability to use a second language while playing games.

# Methodology:

In this project, we will use Unity as a development platform and 
Virtual Reality techniques to build a vocabulary crossword game. 
The project will be divided into 5 steps. 
First step is target population analysis. As described above, 
the population of non-English speakers is large, and the common 
learning obstacle is the amount of vocabulary a person can master. 
According to "The type of vocabulary learning strategies used by ESL students in University Putra Malaysia", categorized 
vocabulary learning strategy helps learners memorize words faster 
by associating and grouping with other words. Thus we will adopt this 
strategy and apply the categorized learning method as baseline. 
Besides, traditional rote learning does not attract people any more, 
learning through VR applications has become a novel efficient approach which 
greatly increases user viscosity. 

Second phase is game designing. From the technical selection perspective,
 we use Oculus as a VR device, thus Unity becomes our first choice as 
 a software platform since it provides Oculus SDKs. From the scientific 
 point of view, we proposed a novel VR learning schema which proceeds 
 learning and gaming alternatively with prompt feedback. We hypothesize 
 this schema will boost the learning efficiency by using our well-designed 
 word selection algorithm. The detailed game design process is as 
 follows(R stands for the robot/system, U represent the user/gamer):
* Step 0 - Before the start of the game, the system equally divides the 
words of the chosen category into 2 parts, then take one 
part (lexicon RL) as lexicon for the robot/machine/system, 
the other part(lexicon UL) as the alternative generative words 
prepared for users/gamers.

* Step1 - R starts first, choosing a word (wordRL)from the alternative lexicon and putting a virtual word on a pre-generated virtual chess board, each character occupying one chess grid.

* Step2 - U choose characters from the set generated by the system(characterUL) from step 0, and put the word on the chessboard, making it cross with any of the words existing on the board.

* Step3 - R selects a vocabulary from the lexicon RL, then puts that word on the chess board, intersecting with any of the vocabularies on board.

* Step4 - Repeat steps 2 and 3 until time is up or the game is over. Here the end of the game means there is no available word that crosses with existing words while on the chess board.

* Step5 - After each round of the game, there is a word learning section, U has to learn 5 words, each word will appear along with the 3D image of it’s physical object to help the user memorize faster and deeper.

* Step6 - Proceed to the next round of the game.

* Step7 - Repeat the above steps for several(2 or 3) times if applicable.

Third phase is the pre-test and post-test designing. 

Before the experiment, game participants will be asked to take a topic-based vocabulary survey. E.g. If a user selects words about food and drinks as the topic, we’ll offer a word familiarity questionnaire about food, including the vocabulary they will learn in the game.

* Design of the questionnaire. For each question, there will be three answers A,B,C indicating different levels of familiarity toward a word(see [details](https://forms.office.com/r/HwakF9qHh9}): 
    * A. Already mastered the word and can explain the meaning.
    * B. Seems familiar but doesn't know the exact meaning.
    * C. Don’t know the word.

* Give weights to different answers. Option A = 2, option B = 1, score for option C is 0. Sum up the scores as the evaluation grade for that participant.

After the experiment, participants will be asked to take a similar [survey](https://forms.office.com/r/Ew4k9kdgWb). 
Compare the pre-test score and post-test score to see how much it increases for different group's participants/users.

Fourth phase is game development, we will use C\# as the programming language, Unity as a software platform and Oculus as the VR device.

Fifth phase are the experiments and surveys, collecting data and evaluation of the result. We will collect participants' scores in the third step as well as their grade wined during the game, then conduct a ANOVA analysis using collected data to see if this VR game practically increases learning efficiency. 


# Game Design:

This VR Crossword Game is divided into five scenes: Start,  Enter ID, Version select,  Learning,  and Gaming.

* Start. The main purpose of setting the game start interface is to make 
the participants adapt to the VR world and VR controller. 
Participants need to use the trigger button in the VR controller 
to click the start button. Participants can also use this interface 
to understand how to use others buttons on the VR controller. 

![Start Scene](./images/startScene.png "Start Scene")

* EnterID. In order to record data conveniently and protect the privacy of users, 
researchers will provide an ID for the participants. 
Participants need to enter the ID into the game to record.

* Version. This game will provide three versions. Participants need to 
choose the version according to the requirements of the researcher.

* Learning. The words in the Learning scene will appear in the game later, 
so participants need to remember all the words in the Learning scene.  
In the current scene, an English explanation of the word appears 
on the panel along with the word, and participants simply drag the 
VR controller up and down to view all the words. 

![Learning Scene](./images/LearningScene.png "Learning Scene")

* In the game scene, there are several reminders(Timer, arrows) 
to remind the participants what need to do.

 ![Game Scene](./images/GameScene.png "Game Scene")
 
 The board starts with the
 word 'egg' in the center, then generates a letter A on the 
 second-to-last row to the far left, with a red arrow indicating 
 which direction to enter the word. After the participants input 
 the current word, they need to click the submit/ Reset button first,
  which is used to upload the current word to the background, and 
  then click the correct or NOT button to check whether the word input 
  is correct. If the input is wrong, they need to re-input the word, 
  otherwise the letters just entered will be fixed on the board. 
  And appear the next word prompt letter and input direction. 
  After the participants input the current word, they need to click 
  the submit/ Reset button first, which is used to upload the current 
  word to the background, and then click the correct or NOT button to 
  check whether the word input is correct. If the input is wrong, 
  they need to re-input the word, otherwise the letters just 
  entered will be fixed on the board.  And appear the next word 
  prompt letter and input direction.  When all the correct words 
  are entered, ***** appears to prompt participants that the 
  experiment has been completed, and the 'Congratulation' appears.
  
  ![Finish Scene](./images/Gamefinish.png "Finish Scene")
  
# Experiments and Results

This experiment adopted the method of control experiment. 
Participants need to take part in two sets of experiments. 
The experiment content of the control group is to complete the 
Crossword puzzle game in the real world. The experimental content 
of the experimental group is to complete game tasks in the VR world. 
Participants need to fill in the post-survey with their preferences 
and ideas after completing the two sets of experiments. 
In the experiment, the two most important variables are the 
independent variable and the dependent variable. In this project, 
the independent variables are mainly divided into the personal 
information of the participants, the way of the game, and the 
environment of the experiment. 
The experiment is mainly divided into three dependent variables. 
The first is the accuracy of the participants during the game, 
and the second is the preference of the participants. The third 
is the time spent participating in each experiment

Independent Variables:

Factor（IV)  | Levels（Test conditions)
------------- | -------------
Personal Information  | Age, English Ability, Favorite Learning method
Game environment  | Real world, VR

Dependent Variables:

ID  | Dependent Variable
------------- | -------------
1  | the accuracy of the participants during the game
2  | the preference of the participants
3  | the time spent participating in each experiment


The number of participants in this experiment is 9 people. Four of them are between 18-25 years old, and five are between 26-30 years old. Among them, 4 women participated in the experiment and 5 men. In addition, more than half of the participants have used VR equipment, but no one has used VR to learn English. According to the pre-survey survey results, 50 percent of participants questioned the effectiveness of using VR to learn English. One of its participants thought that using VR to learn English words is very impractical. After analyzing the results of the survey on the study habits of the participants, it is found that different people use different methods to learn English.
## Procedures:

### Real World Game 
Participants need to complete game tasks in the real world before they start using VR devices to learn English. Participants first need to complete the pre-survey.

![RealWord](./images/RealWorld.png "Real World")

### VR World
Participants need to complete the Pre-survey related to VR game content before starting the game. Participants will wear Oculus Quest VR headset and use the handle to complete the game. Participants need to start the game successfully according to the game prompts and the guidance of the researchers. After starting the game, the researcher will not give participants any more prompts. Participants can only spell the correct words according to the prompts in the game. Participants have only three opportunities to spell each word during the process of participating in the experiment. If all three spellings are wrong, the researcher will directly give the correct answer and list this word as a "unsuccessful" word. After completing all the experiments, participants need to complete a preference survey. This survey will help researchers better understand the participants’ thoughts on the two experiments.

# Result and Analysis

## Compare the difficulty of the two versions
In order to prevent participants from retaining real-world 
word memories when conducting VR experiments, we prepared 
different versions for different experiments. In order to prove 
that the difficulty of the two versions is the same, 
we collected participants' initial awareness of the different versions. 
Among them, the average value of version 1(for VR World) is 5.625. 
The average value of version 2(for Real World) is 5.75. 
The initial familiarity of the users of these two versions is similar, 
so we believe that there is a certain degree of comparability 
between the two versions. The data of the two versions can be compared.

The table of Raw Data is the result about both experiments. P-ID refers to Participant-ID. P-V1 score refers to Pre-Version1 score. Po-V1 score is the abbreviation of Post-Version1 score. P-V2 score is refers to Pre-Version2 score. Po-V2 score is the abbreviation of Post-Version2 score. V1++ and V2 ++ refers to Version 1 improvement and Version 2 improvement.V1 ER refers to  Version Error Rate. 

Raw Data Table

P-ID | P-v1 score | Po-V1 score | v1 ++|V1 ER | P-V2 Score | Po-V2  | V2 ER | v2 ++
--- | ---------- | ----------|------ | -----|----------- | -------| ------| -----
  1 | 8 | 10 | 2|0.267| 28 | 10 | 0.4|2
  2 | 4 | 7 | 3|0.267 |6 | 7 | 0.33 |1
  3 | 4 | 10 | 6| 0.4 | 4 | 10 | 0.33|6
  4 | 9 | 10 | 1|0.4 | 8 | 10 | 0.2 | 2
  5 | 7 | 10 | 3|0 | 5 | 10 | 0|5
  6 | 6 | 8 | 2|0.4 | 4 | 8 | 0.467 |4
  7 | 4 | 9 | 5|0.6 | 4 | 9 | 0.2|5
  8 | 4 | 9 | 5|0.2 | 6 | 10 | 0.133|4
  
![CompareV1V2](./images/Compare.png "CompareV1V2")
  

  
## Control Group

* Compare Pre-Survey and Post-Survery of Version 1. 
According to the above data, it can be found that most of the participants can recite words very well, although they still use the traditional methods. In addition, the basic English skills of Participants No. 2 and No. 3 are not very strong, but they can still get a high score in the end by using traditional methods. It is obvious that traditional methods have far-reaching influence on students. Although some participants said that using traditional methods does make them feel a lot of pressure, but it is indeed a very effective method. They are also always looking for better ways to recite words.
* Analysis the Error Rate.
According to the results of Version 2 error rate, the error rate of a lot of participants is not high, which is the same as the results of pre-survey and Post-survey. Participant No. 5 has an error rate of 0. This participant has a very strong basic learning ability. The traditional learning method is very suitable for him.

## Experimental Group

In this study, we use ANOVA test(Analysis of variance) to evaluate 
the improvement of effectiveness. 
We pasteurized the familiarity to words using the word pre-survey score. 
Our null hypothesis is that VR crossword game does not improve users' 
familiarity with words, i.e. VR crossword game can not help users 
memorize English vocabularies. There are two groups with 16 result values,
 thus the first degree of freedom is 1 and second degree of 
 freedom is 14. Sum of square within groups is 39 while the sum of 
 square is 49. Based on the previous data and F score 
 algorithm. Assume p<0.05, check F distribution table F(1,14)=4.6. The F score is 17.59>F(1,14), thus we reject the null hypothesis. 
 The above data and analysis proved that VR technique can effectively 
 augment the memory ability of users.
 
 ## Compare results of two groups
 

 Based on the score improvement made by two different groups,
  we analyzed the validity of VR crossword game compared to real
   world physical game. 
   
   ![CompareV1](./images/CompareVersion1.png "CompareV1")
   
   Apply the above ANOVA analysis method 
   on improvement data. Our hypothesis is that the English word 
   memorize ability improvement made through VR crossword game 
   is less than it made through playing physical world game.Sum of 
   square within groups is 43.75 while the sum of square is 44. 
   Assume p<0.05, the F score is 0.08<F(1,14), thus we accept the 
   null hypothesis. Which indicates that the real world crossword 
   game outperformed the VR application we developed.
   
   ![CompareV2](./images/CompareVersion2.png "CompareV2")
    
    
By comparing VR test and paper test, the error rate of VR test is 0.316, 
while that of paper test is 0.2575. It can be obviously found that paper 
test has a lower error rate. Here's what we can guess about this result: 
People doing VR experiments will make mistakes because they are 
unfamiliar with the operation. The operation of VR is complicated, 
which reduces the efficiency of learning words. When people play 
VR for the first time, they will feel excited or nervous. We guess 
that both positive and negative emotions will affect the efficiency 
of learning words. In the paper test, the participants were allowed 
to spell words out of order, but in the VR test, the words had to be 
spelled backwards. We speculated that this rule might also affect word 
memory or spelling efficiency. We found that people are more used to 
this more traditional method of memorizing words. 
   
# Conclusion:

In this experiment, we found that Crossword game is very effective 
in learning English words, which can greatly increase the fun of the 
game and reduce people's anxiety and boredom when reciting words. 
Experimental data show that although VR can improve the efficiency 
of memorizing words, the efficiency improvement it brings is not as 
high as the control group, that is, the paper test. 
People more bias from the traditional method of memorizing words, 
we speculate that the test on the paper, will reduce the invalid 
behavior of participants, comparing the VR group, test group will 
not be used on the paper the uncomfortable feeling when VR or excitement, 
so a paper test group efficiency will be higher than the VR group, 
considering the VR group need again with handle in VR games move the 
letters, Thus, participants' emotions may fluctuate due to inefficient 
letter movement rates. In addition, through the experimental data, 
we found that the paper tests the small errors, we guess is we will 
not rule the experimenter spelling words on the paper order, 
but in VR, participants in the input sequence is must, in turn, 
input the correct sequence of letters, so this is not convenience 
may results in the decrease of the efficiency of the participants 
spelling words. Through the questionnaire survey,
 we found that people are not averse to using VR as a new way to 
 memorize words, but almost no one will actively choose to use VR 
 to memorize words, and people still prefer to use more traditional 
 methods to memorize words. 
 This is similar to the experimental results. 
 Our conclusion is that VR is an effective new way to memorize words, 
 but the efficiency improvement is not as high as the control group.

# Partial References:
[1\] Xiao Jing and Wang Jing. 2019. Effectiveness of mobile learning in college 
English vocabulary teaching under multimodal environment. 
In Proceedings of the 10th International Conference on E-Education, 
E- Business, E-Management and E-Learning Association for 
Computing Machinery, New York, NY, USA, 93–97. 
DOI: https://doi.org/10.1145/3306500.3306558

[2\] Shabaneh, Yasmin | Farrah, Mohammed. (2019). THE EFFECT OF GAMES ON VOCABULARY RETENTION. Indonesian Journal of Learning and Instruction. 2. 79-90. 10.25134/Ijli. v2i01.1687.

[3\] Li-Jie Chang, Jie-Chi Yang | Fu-Yun Yu (2003) Development and evaluation of multiple competitive activities in a synchronous quiz game system, Innovations in Education and Teaching International, 40:1, 16- 26, DOI: 10.1080/1355800032000038840

[4\] Chiu-Pin Lin, Shelley S.-C. Young, Hui-Chung Hung, and Yi-Chen Lin. 2007. Implementation of the scrabble game on the mobile devices to increase english vocabulary acquisition. In Proceedings of the 8th iternational conference on Computer supported collaborative learning International Society of the Learning Sciences, 441–443.

[5\] M. Billinghurst, J. LaViola and A. Lecuyer. 2012. IEEE Symposiumon 3D User Interfaces Proceedings (Eds.). Piscataway: IEEE.https://dblp.org/db/conf/3dui/3dui2012.html\

# Links:
* [Github Repository](https://github.com/csu-hci-projects/An-efficient-language-learning-method-VR-vocabulary-crossword-puzzle-game)
* [Term Paper in Overleaf](https://www.overleaf.com/project/61845a4826f07579b02cdf1d)
* [Checkpoint 1 Video](https://youtu.be/_x09YcM5B1A)
* [Checkpoint 2 Video](https://www.youtube.com/watch?v=tcR1p2ma5-w|ab_channel=JacindaLi)

