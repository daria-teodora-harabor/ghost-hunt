# tinyMMLU — 100 items

Source: `tinyBenchmarks/tinyMMLU` config=`all` split=`test` (as loaded by lm-eval).

Correct choice marked with `*`. tinyMMLU's `input_formatted` already embeds the 5 dev-split few-shot examples, so each block below is the full prompt the model sees; the scored question is the LAST one in the block.

### 0. [high_school_statistics]

```
The following are multiple choice questions (with answers) about high school statistics.

Which of the following is a correct statement about correlation?
A. If the slope of the regression line is exactly 1, then the correlation is exactly 1.
B. If the correlation is 0, then the slope of the regression line is undefined.
C. Switching which variable is called x and which is called y changes the sign of the correlation.
D. The correlation r is equal to the slope of the regression line when z-scores for the y-variable are plotted against z-scores for the x-variable.
Answer: D

Suppose X and Y are random variables with E(X) = 37, var(X) = 5, E(Y) = 62, and var(Y) = 12. What are the expected value and variance of the random variable X + Y?
A. E(X + Y) = 99, var(X + Y) = 8.5
B. E(X + Y) = 99, var(X + Y) = 13
C. E(X + Y) = 99, var(X + Y) = 17
D. There is insufficient information to answer this question.
Answer: D

After a frost warning was issued, the owner of a large orange grove asked his workers to spray all his trees with water. The water was supposed to freeze and form a protective covering of ice around the orange blossom. Nevertheless, the owner suspected that some trees suffered considerable damage due to the frost. To estimate the proportion of trees that suffered more than 50 percent damage due to the frost, he took a random sample of 100 trees from his grove. What is the response variable in this experiment?
A. The proportion of trees that suffered more than 50 percent damage due to frost.
B. The number of trees affected by the frost.
C. The number of trees sampled from the grove.
D. For each sampled tree, whether it suffered more than 50 percent damage or at most 50 percent damage.
Answer: D

A new smartwatch is manufactured in one part of a factory, then secured for shipping in another, independent part of the factory. The weight of the smartwatch has a mean of 62 grams and a standard deviation of 1.0 grams. The weight of the packaging (box, user's guide, bubble wrap, etc.) has a mean of 456 grams and a standard deviation of 6 grams. Together, the distribution of the weight of the smartwatch and its packaging would have the following mean and standard deviation:
A. Mean 518 grams; standard deviation 7.0 grams
B. Mean 518 grams; standard deviation 3.5 grams
C. Mean 518 grams; standard deviation 6.1 grams
D. Mean 394 grams; standard deviation 6.1 grams
Answer: C

Which of the following sets has the smallest standard deviation? Which has the largest?
I: {1,2,3}
II: {-10,10}
III: {100}
A. I, II
B. II, III
C. III, I
D. III, II
Answer: D

The number of days it takes to build a new house has a variance of 386. A sample of 40 new homes shows an average building time of 83 days. With what confidence can we assert that the average building time for a new house is between 80 and 90 days?
A. 15.4%
B. 17.8%
C. 20.0%
D. 82.1%
Answer:
```
Answer: **D**

### 1. [nutrition]

```
The following are multiple choice questions (with answers) about nutrition.

In a cohort study, the risk ratio of developing diabetes was 0.86 when comparing consumers of tea (the exposed) to those who did not drink tea (the unexposed). Which one statement is correct (according to knowledge in 2020)?
A. The tea drinkers have lower risk of developing diabetes.
B. The tea drinkers have higher risk of developing diabetes.
C. Based on the information given we cannot tell if the observed difference in disease risk is the result of chance.
D. The risk ratio is close to the value one, so there is no difference in disease risk between the two groups.
Answer: C

Which of the following statements is correct (according to knowledge in 2020)?
A. Consumers with phenylketonuria must avoid the consumption of the sweetener aspartame
B. Consumers with phenylketonuria must avoid the consumption of the sweetener saccharin
C. Consumers with phenylketonuria must avoid the consumption of the sweetener sucralose
D. Consumers with phenylketonuria must avoid the consumption of the sweetener acesulfame K
Answer: A

Which of the following is the most plausible explanation for the protective effect of dietary fibre against cancer of the colon, as of 2020?
A. Propionic acid, formed during colonic fibre fermentation inhibits liver fatty acid synthesis
B. Butyric acid, formed during colonic fibre fermentation stimulates "silencing" of the SLC5A8 tumour suppressor gene
C. None of these options are correct
D. Butyric acid, formed during colonic fibre fermentation stimulates anti-oxidant defences in the colon
Answer: D

Which of the following statements about iodine is correct, as of 2020?
A. 50% of adults consume iodine at levels below the RNI
B. Dairy products are a poor source of iodine
C. The iodine content of organic milk is generally lower that the level in non-organic milk
D. UK dietary reference values recommend an increase in iodine intake in pregnancy
Answer: C

What is the first-line drug for patients with type 2 diabetes and obesity, as of 2020?
A. Acarbose
B. Metformin
C. Sulphonylureas
D. Insulin
Answer: B

According to trials on diabetes prevention, high-risk individuals can reduce their risk to develop diabetes by doing the following:
A. Eating a very low carbohydrate diet
B. Consuming a diet high in monounsaturated fats
C. Losing 5-7% of body weight through a hypocaloric low fat diet and 30 minutes of daily activity
D. Initiating metformin 850 mg BID and practicing daily vigorous exercise
Answer:
```
Answer: **C**

### 2. [clinical_knowledge]

```
The following are multiple choice questions (with answers) about clinical knowledge.

The energy for all forms of muscle contraction is provided by:
A. ATP.
B. ADP.
C. phosphocreatine.
D. oxidative phosphorylation.
Answer: A

What is the difference between a male and a female catheter?
A. Male and female catheters are different colours.
B. Male catheters are longer than female catheters.
C. Male catheters are bigger than female catheters.
D. Female catheters are longer than male catheters.
Answer: B

In the assessment of the hand function which of the following is true?
A. Abduction of the thumb is supplied by spinal root T2
B. Opposition of the thumb by opponens policis is supplied by spinal root T1
C. Finger adduction is supplied by the median nerve
D. Finger abduction is mediated by the palmar interossei
Answer: B

How many attempts should you make to cannulate a patient before passing the job on to a senior colleague, according to the medical knowledge of 2020?
A. 4
B. 3
C. 2
D. 1
Answer: C

Glycolysis is the name given to the pathway involving the conversion of:
A. glycogen to glucose-1-phosphate.
B. glycogen or glucose to fructose.
C. glycogen or glucose to pyruvate or lactate.
D. glycogen or glucose to pyruvate or acetyl CoA.
Answer: C

Which of the following is true regarding the complications of a large goitre?
A. It causes stridor due to compression of the oesophagus
B. Dysphagia is an indication for surgery
C. It causes hoarseness due to compression of the trachea
D. It causes a more florid thyrotoxicosis than a smaller goitre
Answer:
```
Answer: **B**

### 3. [nutrition]

```
The following are multiple choice questions (with answers) about nutrition.

In a cohort study, the risk ratio of developing diabetes was 0.86 when comparing consumers of tea (the exposed) to those who did not drink tea (the unexposed). Which one statement is correct (according to knowledge in 2020)?
A. The tea drinkers have lower risk of developing diabetes.
B. The tea drinkers have higher risk of developing diabetes.
C. Based on the information given we cannot tell if the observed difference in disease risk is the result of chance.
D. The risk ratio is close to the value one, so there is no difference in disease risk between the two groups.
Answer: C

Which of the following statements is correct (according to knowledge in 2020)?
A. Consumers with phenylketonuria must avoid the consumption of the sweetener aspartame
B. Consumers with phenylketonuria must avoid the consumption of the sweetener saccharin
C. Consumers with phenylketonuria must avoid the consumption of the sweetener sucralose
D. Consumers with phenylketonuria must avoid the consumption of the sweetener acesulfame K
Answer: A

Which of the following is the most plausible explanation for the protective effect of dietary fibre against cancer of the colon, as of 2020?
A. Propionic acid, formed during colonic fibre fermentation inhibits liver fatty acid synthesis
B. Butyric acid, formed during colonic fibre fermentation stimulates "silencing" of the SLC5A8 tumour suppressor gene
C. None of these options are correct
D. Butyric acid, formed during colonic fibre fermentation stimulates anti-oxidant defences in the colon
Answer: D

Which of the following statements about iodine is correct, as of 2020?
A. 50% of adults consume iodine at levels below the RNI
B. Dairy products are a poor source of iodine
C. The iodine content of organic milk is generally lower that the level in non-organic milk
D. UK dietary reference values recommend an increase in iodine intake in pregnancy
Answer: C

What is the first-line drug for patients with type 2 diabetes and obesity, as of 2020?
A. Acarbose
B. Metformin
C. Sulphonylureas
D. Insulin
Answer: B

Doses of substances used in animal toxicity studies are usually many times greater than those experienced by people because:
A. Animals are generally much less sensitive than people
B. Such tests can detect only very large risks because of the fact that only small numbers of animals can be used
C. It is difficult in such studies to determine whether the observed toxic effects were caused by the substance
D. It is the only way to ensure results are applicable to people
Answer:
```
Answer: **B**

### 4. [high_school_mathematics]

```
The following are multiple choice questions (with answers) about high school mathematics.

Joe was in charge of lights for a dance. The red light blinks every two seconds, the yellow light every three seconds, and the blue light every five seconds. If we include the very beginning and very end of the dance, how many times during a seven minute dance will all the lights come on at the same time? (Assume that all three lights blink simultaneously at the very beginning of the dance.)
A. 3
B. 15
C. 6
D. 5
Answer: B

Five thousand dollars compounded annually at an $x\%$ interest rate takes six years to double. At the same interest rate, how many years will it take $\$300$ to grow to $\$9600$?
A. 12
B. 1
C. 30
D. 5
Answer: C

The variable $x$ varies directly as the square of $y$, and $y$ varies directly as the cube of $z$. If $x$ equals $-16$ when $z$ equals 2, what is the value of $x$ when $z$ equals $\frac{1}{2}$?
A. -1
B. 16
C. -\frac{1}{256}
D. \frac{1}{16}
Answer: C

Simplify and write the result with a rational denominator: $$\sqrt{\sqrt[3]{\sqrt{\frac{1}{729}}}}$$
A. \frac{3\sqrt{3}}{3}
B. \frac{1}{3}
C. \sqrt{3}
D. \frac{\sqrt{3}}{3}
Answer: D

Ten students take a biology test and receive the following scores: 45, 55, 50, 70, 65, 80, 40, 90, 70, 85. What is the mean of the students’ test scores?
A. 55
B. 60
C. 62
D. 65
Answer: D

Rowena can paint a room in $14$ hours, while Ruby can paint it in $6$ hours. If Rowena paints for $x$ hours and Ruby paints for $y$ hours, they will finish half of the painting, while if Rowena paints for $y$ hours and Ruby paints for $x$ hours they will paint the whole room. Find the ordered pair $(x,y)$.
A. (\frac{11}{10}, \frac{11}{10})
B. (\frac{231}{20}, \frac{21}{20})
C. (\frac{231}{40}, \frac{21}{40})
D. (1,1)
Answer:
```
Answer: **C**

### 5. [moral_disputes]

```
The following are multiple choice questions (with answers) about moral disputes.

According to Metz, what is wrong with consequentialist arguments against capital punishment based on African values?
A. It is unclear as of yet whether or not capital punishment deters harm to the community.
B. It is unclear as of yet whether or not capital punishment deters harm to any individuals.
C. Consequentialism is not supported by African values.
D. Even though consequentialism is supported by African values, no consequentialist arguments framed in terms of African values have been offered.
Answer: A

A fertilized ovum is also known as
A. a zygote.
B. an embryo.
C. a viability.
D. a blastocyst.
Answer: A

Which of the following is an intrinsic good, according to Singer?
A. being in the upper class
B. the ability to learn and solve problems
C. being more beautiful than most of the population
D. being shorter than average
Answer: B

Pence compares six different cases of reproduction, from natural twinning to SCNT. What conclusion does he draw from this comparison?
A. SCNT is not a different kind of reproduction because there are no morally relevant differences between it and other permissible means of reproduction.
B. Because there is a low risk of harm for natural twinning, there will be a low risk of harm for SCNT.
C. both A and B
D. neither A nor B
Answer: A

Baron admits that the versions of the ticking bomb hypothetical she discusses are "stunningly stupid," but she claims this is actually evidence of
A. the stupidity of most traditional philosophical examples.
B. a general lack of intelligence among people with advanced degrees.
C. the wrongness of torture.
D. the readiness on the part of many intelligent people to see torture as the best solution to deal with terrorism.
Answer: D

Which of the following statements would Macedo disagree with?
A. States should give special weight to their own citizens.
B. Immigration policies can have powerful economic effects.
C. Self-governance is morally irrelevant to the immigration debate.
D. There are reasons to think U.S. immigration policy has had detrimental effects on income distribution among citizens.
Answer:
```
Answer: **C**

### 6. [professional_medicine]

```
The following are multiple choice questions (with answers) about professional medicine.

A 42-year-old man comes to the office for preoperative evaluation prior to undergoing adrenalectomy scheduled in 2 weeks. One month ago, he received care in the emergency department for pain over his right flank following a motor vehicle collision. At that time, blood pressure was 160/100 mm Hg and CT scan of the abdomen showed an incidental 10-cm left adrenal mass. Results of laboratory studies, including complete blood count, serum electrolyte concentrations, and liver function tests, were within the reference ranges. The patient otherwise had been healthy and had never been told that he had elevated blood pressure. He takes no medications. A follow-up visit in the office 2 weeks ago disclosed elevated urinary normetanephrine and metanephrine and plasma aldosterone concentrations. The patient was referred to a surgeon, who recommended the adrenalectomy. Today, vital signs are temperature 36.6°C (97.9°F), pulse 100/min, respirations 14/min, and blood pressure 170/95 mm Hg. Physical examination discloses no significant findings. Initial preoperative preparation should include treatment with which of the following?
A. Labetalol
B. A loading dose of potassium chloride
C. Nifedipine
D. Phenoxybenzamine
Answer: D

A 36-year-old male presents to the office with a 3-week history of low back pain. He denies any recent trauma but says that he climbs in and out of his truck numerous times a day for his job. Examination of the patient in the prone position reveals a deep sacral sulcus on the left, a posterior inferior lateral angle on the right, and a lumbosacral junction that springs freely on compression. The most likely diagnosis is
A. left-on-left sacral torsion
B. left-on-right sacral torsion
C. right unilateral sacral flexion
D. right-on-right sacral torsion
Answer: D

A previously healthy 32-year-old woman comes to the physician 8 months after her husband was killed in a car crash. Since that time, she has had a decreased appetite and difficulty falling asleep. She states that she is often sad and cries frequently. She has been rechecking the door lock five times before leaving her house and has to count exactly five pieces of toilet paper before she uses it. She says that she has always been a perfectionist but these urges and rituals are new. Pharmacotherapy should be targeted to which of the following neurotransmitters?
A. Dopamine
B. Glutamate
C. Norepinephrine
D. Serotonin
Answer: D

A 44-year-old man comes to the office because of a 3-day history of sore throat, nonproductive cough, runny nose, and frontal headache. He says the headache is worse in the morning and ibuprofen does provide some relief. He has not had shortness of breath. Medical history is unremarkable. He takes no medications other than the ibuprofen for pain. Vital signs are temperature 37.4°C (99.4°F), pulse 88/min, respirations 18/min, and blood pressure 120/84 mm Hg. Examination of the nares shows erythematous mucous membranes. Examination of the throat shows erythema and follicular lymphoid hyperplasia on the posterior oropharynx. There is no palpable cervical adenopathy. Lungs are clear to auscultation. Which of the following is the most likely cause of this patient's symptoms?
A. Allergic rhinitis
B. Epstein-Barr virus
C. Mycoplasma pneumoniae
D. Rhinovirus
Answer: D

A 22-year-old male marathon runner presents to the office with the complaint of right-sided rib pain when he runs long distances. Physical examination reveals normal heart and lung findings and an exhalation dysfunction at ribs 4-5 on the right. Which of the following muscles or muscle groups will be most useful in correcting this dysfunction utilizing a direct method?
A. anterior scalene
B. latissimus dorsi
C. pectoralis minor
D. quadratus lumborum
Answer: C

A 17-year-old girl is brought to the physician by her mother because she has not had a menstrual period for 6 months. The patient is unconcerned about the lack of menses. Menarche occurred at the age of 12 years, and menses had occurred at regular 28-day intervals until they became irregular 1 year ago. She is a member of her high school gymnastics team. She appears emaciated. She is 163 cm (5 ft 4 in) tall and weighs 40 kg (88 lb); BMI is 15 kg/m2 . Her pulse is 54/min, and blood pressure is 80/50 mm Hg. Which of the following is the most likely cause of this patient's amenorrhea?
A. Hyperthyroidism
B. Hypogonadotropic hypogonadism
C. Hypothyroidism
D. Polycystic ovarian syndrome
Answer:
```
Answer: **B**

### 7. [prehistory]

```
The following are multiple choice questions (with answers) about prehistory.

The great Mayan king Pacal built temples in the city of Palenque in order to:
A. satisfy the powerful Mayan astronomer priests.
B. display his generosity to the common people, since they were allowed to live in the temples.
C. frighten away enemies, in particular the Spaniards.
D. legitimize his kingship, since his father was not royal.
Answer: D

According to Timothy Pauketat, the evidence for social stratification and political power at Cahokia suggests:
A. a center of Mississippian civilization with conditions similar to the rise of early states.
B. the limitations of authority in a Native American society of egalitarian foragers.
C. a simple chiefdom or perhaps a complex chiefdom had evolved by A.D. 1500.
D. a center of Mississippian civilization with conditions similar to societies on the Northwest Coast of North America.
Answer: A

Researchers now believe that the decline of the Maya was caused chiefly by:
A. a cataclysm of some kind, such as an earthquake, volcano, or tsunami.
B. ecological degradation resulting from slash-and-burn farming techniques.
C. endless wars between neighboring Mayan city-states.
D. practices of interbreeding that led to a steep rise in congenital disorders.
Answer: B

Recent research on hominid species dating from the Middle Pliocene indicates there was (as of 2020):
A. a great amount of species diversity, or a single species that exhibited a lot of diversity.
B. very little species diversity during this period and very few hominids.
C. decreased species diversity due to a prolonged ice age followed by a severe drought.
D. decreased species diversity but increased numbers of hammerstones and flakes, indicating stone tool manufacture.
Answer: A

What is the approximate mean cranial capacity of Homo erectus?
A. under 650 cc
B. about 800 cc
C. just under 1000 cc
D. 1200 cc
Answer: C

The Yang-shao culture gave way to the Lung-Shan sometime after:
A. 6,000 B.P.
B. 5,000 B.P.
C. 4,000 B.P.
D. 3,000 B.P.
Answer:
```
Answer: **B**

### 8. [business_ethics]

```
The following are multiple choice questions (with answers) about business ethics.

Beyond the business case for engaging in CSR there are a number of moral arguments relating to: negative _______, the _______that corporations possess and the ________ of business and society.
A. Externalities, Power, Independence
B. Publicity, Insubstantial resources, Mutual dependence
C. Publicity, Power, Independence
D. Externalities, Power, Mutual dependence
Answer: D

_______ is the direct attempt to formally or informally manage ethical issues or problems, through specific policies, practices and programmes.
A. Corporate social responsibility
B. Business ethics management
C. Sustainability
D. Environmental management
Answer: B

To ensure the independence of the non-executive board members, they are a number of steps which can be taken, which include non-executives being drawn from _______ the company, being appointed for a _________ time period as well as being appointed _________.
A. Outside, Limited, Independently
B. Inside, Limited, Intermittently
C. Outside, Unlimited, Intermittently
D. Inside, Unlimited, Independently
Answer: A

Three contrasting tactics that CSO's can engage in to meet their aims are ________ which typically involves research and communication, ________, which may involve physically attacking a company's operations or ________, often involving some form of _______.
A. Non-violent direct action, Violent direct action, Indirect action, Boycott
B. Indirect action, Instrumental action, Non-violent direct action, Information campaign
C. Indirect action, Violent direct action, Non-violent direct-action Boycott
D. Non-violent direct action, Instrumental action, Indirect action, Information campaign
Answer: C

In contrast to _______, _______ aim to reward favourable behaviour by companies. The success of such campaigns have been heightened through the use of ___________, which allow campaigns to facilitate the company in achieving _________ .
A. Buycotts, Boycotts, Blockchain technology, Charitable donations
B. Buycotts, Boycotts, Digital technology, Increased Sales
C. Boycotts, Buyalls, Blockchain technology, Charitable donations
D. Boycotts, Buycotts, Digital technology, Increased Sales
Answer: D

Cultural homogenization.
A. 1,3,4
B. 1,2,3
C. 1,2,3,4
D. 2,3,4
Answer:
```
Answer: **B**

### 9. [philosophy]

```
The following are multiple choice questions (with answers) about philosophy.

Psychological egoism is:
A. an ethical theory about how we ought to behave.
B. a generalization concerning the way people tend to behave.
C. a claim about human nature and the ways people are capable of behaving.
D. none of the above.
Answer: C

According to Moore’s “ideal utilitarianism,” the right action is the one that brings about the greatest amount of:
A. pleasure.
B. happiness.
C. good.
D. virtue.
Answer: C

According to d'Holbach, people always act according to _____.
A. free choices
B. dictates of the soul
C. necessary natural laws
D. undetermined will
Answer: C

Before Tolstoy's Christian conversion, what was his perspective on the meaning of life?
A. optimist
B. satisfied
C. nominally religious
D. pessimist
Answer: D

The study of reality in the broadest sense, an inquiry into the elemental nature of the universe and the things in it, is known as _____.
A. metaphysics
B. epistemology
C. quantum physics
D. axiology
Answer: A

Of the ways of escaping moral arguments that he discusses, Hare claims:
A. none of them are successful.
B. some, but not all of them are successful.
C. all are entirely successful.
D. all might be successful, but at a price.
Answer:
```
Answer: **D**

### 10. [professional_psychology]

```
The following are multiple choice questions (with answers) about professional psychology.

One of your therapy clients asks your advice about a good weight- reduction program. You have investigated the programs in the community and are enrolled in the one you consider the best. This program offers a $50 bonus to its patrons for each new person they bring into the program. Under these circumstances, your most appropriate response would be to
A. tell your client the pros and cons of each program you know about except for the one in which you are enrolled
B. recommend to your client the program in which you are enrolled and explain the $50 bonus you will receive
C. recommend to your client the program in which you are enrolled and offer to have the $50 bonus credited to your client's account in the program
D. tell your client the pros and cons of each program you know about, but do not claim the $50 bonus if your client enrolls in your program
Answer: D

There are three ways to measure the Central Tendency: the Mean, the Median and the Mode. From your knowledge about them, what is the mode?
A. less sensitive to extreme scores than the mean
B. more useful for skewed distributions
C. sensitive to extreme values and highly skewed distributions
D. the most frequently occurring number
Answer: D

In terms of Hofstede’s (1980) five cultural dimensions, the United States scores at the top of the scale on:
A. individualism.
B. individualism and power distance.
C. power distance and masculinity.
D. uncertainty avoidance.
Answer: A

Carl Jung believed that a client's transference:
A. is a fantasy that distracts the client from reality.
B. represents “mixed feelings” toward the therapist. 
C. "is a form of ""acting out."""
D. reflects the client’s personal and collective unconscious.
Answer: D

In the construction of a multiple regression equation for purposes of prediction, the optimal combination of measures is one in which the predictors
A. are uncorrelated with each other but are moderately correlated with the criterion
B. have low correlations with each other and low correlations with the criterion
C. are highly intercorrelated with each other and moderately correlated with the criterion
D. have low correlations with the criterion bur are moderately correlated with each other
Answer: A

An employee of large manufacturing firm finds the work dull, and pursues the job only for the money. The employee performs minimally, and is not motivated so produce highest quantity or better quality. According to F. Herzberg's two-factor theory, the best way 10 increase this worker's self-motivation would be to
A. offer a raise and incentive pay for greater productivity
B. redesign the job to provide a challenge and a sense of accomplishment
C. increase opportunities for informed social interaction
D. provide the employee with a job tide carrying greater prestige
Answer:
```
Answer: **B**

### 11. [professional_accounting]

```
The following are multiple choice questions (with answers) about professional accounting.

Box a nongovernmental not-for-profit organization had the following transactions during the year: Proceeds from sale of investments $80000 Purchase of property plant and equipment $10000 Proceeds from long-term debt $100000 Loss on sale of investment $5000 What amount should be reported as net cash provided by financing activities in Box's statement of cash flows?
A. $70,000
B. $75,000
C. $80,000
D. 100000
Answer: D

One hundred years ago, your great-great-grandmother invested $100 at 5% yearly interest. What is the investment worth today?
A. $13,000
B. $600
C. $15,000
D. $28,000
Answer: A

Krete is an unmarried taxpayer with income exclusively from wages. By December 31, year 1, Krete's employer has withheld $16,000 in federal income taxes and Krete has made no estimated tax payments. On April 15, year 2, Krete timely filed for an extension request to file her individual tax return, and paid $300 of additional taxes. Krete's year 1 tax liability was $16,500 when she timely filed her return on April 30, year 2, and paid the remaining tax liability balance. What amount would be subject to the penalty for underpayment of estimated taxes?
A. $0
B. $500
C. $1,650
D. $16,500
Answer: A

On January 1, year 1, Alpha Co. signed an annual maintenance agreement with a software provider for $15,000 and the maintenance period begins on March 1, year 2. Alpha also incurred $5,000 of costs on January 1, year 1, related to software modification requests that will increase the functionality of the software. Alpha depreciates and amortizes its computer and software assets over five years using the straight-line method. What amount is the total expense that Alpha should recognize related to the maintenance agreement and the software modifications for the year ended December 31, year 1?
A. $5,000
B. $13,500
C. $16,000
D. $20,000
Answer: B

An auditor traces the serial numbers on equipment to a nonissuer’s subledger. Which of the following management assertions is supported by this test?
A. Valuation and allocation
B. Completeness
C. Rights and obligations
D. Presentation and disclosure
Answer: B

In an e-commerce environment that requires that the information technology (IT) system be available on a continuous basis, more emphasis will be placed on which of the following aspects of the planning than in a traditional organization?
A. Maintain appropriate written source documents so the data can be re-entered if it is lost or compromised
B. Maintain redundant systems for instant availability to assure the flow of transactions
C. Review additional expenses to obtain the required amount of business interruption insurance coverage for the organization
D. Assure that appropriate data backups are stored in an off-site location
Answer:
```
Answer: **B**

### 12. [high_school_physics]

```
The following are multiple choice questions (with answers) about high school physics.

Which of the following conditions will ensure that angular momentum is conserved? I. Conservation of linear momentum II. Zero net external force III. Zero net external torque
A. I and II only
B. I and III only
C. II and III only
D. III only
Answer: D

A pipe full of air is closed at one end. A standing wave is produced in the pipe, causing the pipe to sound a note. Which of the following is a correct statement about the wave’s properties at the closed end of the pipe?
A. The pressure is at a node, but the particle displacement is at an antinode.
B. The pressure is at an antinode, but the particle displacement is at a node.
C. The pressure and the particle displacement are both at nodes.
D. The pressure and the particle displacement are both at antinodes.
Answer: B

A photocell of work function ϕ = 2eV is connected to a resistor in series. Light of frequency f = 1 × 10^15 Hz hits a metal plate of the photocell. If the power of the light is P = 100 W, what is the current through the resistor?
A. 2:00 AM
B. 6:00 AM
C. 12:00 AM
D. 24 A
Answer: D

A microwave oven is connected to an outlet, 120 V, and draws a current of 2 amps. At what rate is energy being used by the microwave oven?
A. 10 W
B. 30 W
C. 60 W
D. 240 W
Answer: D

A point charge, Q = +1 mC, is fixed at the origin. How much work is required to move a charge, Q = +8 µC, from the point (0, 4 meters) to the point (3 meters, 0)?
A. 3.5 J
B. 6.0 J
C. 22.5 J
D. 40 J
Answer: B

The Space Shuttle orbits 300 km above the Earth’s surface; the Earth’s radius is 6400 km. What is the acceleration due to Earth’s gravity experienced by the Space Shuttle?
A. 4.9 m/s^2
B. 8.9 m/s^2
C. 9.8 m/s^2
D. 10.8 m/s^2
Answer:
```
Answer: **B**

### 13. [abstract_algebra]

```
The following are multiple choice questions (with answers) about abstract algebra.

Find all c in Z_3 such that Z_3[x]/(x^2 + c) is a field.
A. 0
B. 1
C. 2
D. 3
Answer: B

Statement 1 | If aH is an element of a factor group, then |aH| divides |a|. Statement 2 | If H and K are subgroups of G then HK is a subgroup of G.
A. True, True
B. False, False
C. True, False
D. False, True
Answer: B

Statement 1 | Every element of a group generates a cyclic subgroup of the group. Statement 2 | The symmetric group S_10 has 10 elements.
A. True, True
B. False, False
C. True, False
D. False, True
Answer: C

Statement 1| Every function from a finite set onto itself must be one to one. Statement 2 | Every subgroup of an abelian group is abelian.
A. True, True
B. False, False
C. True, False
D. False, True
Answer: A

Find the characteristic of the ring 2Z.
A. 0
B. 3
C. 12
D. 30
Answer: A

Statement 1 | The image of a group of 6 elements under a homomorphism may have 12 elements. Statement 2 | There is a homomorphism of some group of 6 elements into some group of 12 elements.
A. True, True
B. False, False
C. True, False
D. False, True
Answer:
```
Answer: **D**

### 14. [anatomy]

```
The following are multiple choice questions (with answers) about anatomy.

What is the embryological origin of the hyoid bone?
A. The first pharyngeal arch
B. The first and second pharyngeal arches
C. The second pharyngeal arch
D. The second and third pharyngeal arches
Answer: D

Which of these branches of the trigeminal nerve contain somatic motor processes?
A. The supraorbital nerve
B. The infraorbital nerve
C. The mental nerve
D. None of the above
Answer: D

The pleura
A. have no sensory innervation.
B. are separated by a 2 mm space.
C. extend into the neck.
D. are composed of respiratory epithelium.
Answer: C

In Angle's Class II Div 2 occlusion there is
A. excess overbite of the upper lateral incisors.
B. negative overjet of the upper central incisors.
C. excess overjet of the upper lateral incisors.
D. excess overjet of the upper central incisors.
Answer: C

Which of the following is the body cavity that contains the pituitary gland?
A. Abdominal
B. Cranial
C. Pleural
D. Spinal
Answer: B

At which of the following locations does bile enter the digestive tract?
A. Gastroesophageal sphincter
B. Duodenum
C. Ileocecum
D. Jejunum
Answer:
```
Answer: **B**

### 15. [high_school_world_history]

```
The following are multiple choice questions (with answers) about high school world history.

This question refers to the following information.
He contains all works and desires and all perfumes and all tastes. He enfolds the whole universe and in silence is loving to all. This is the Spirit that is in my heart, this is Brahman. To him I shall come when I go beyond this life, and to him will come he who has faith and doubts not.
—The Upanishads, India, c. 1000 BCE
To which religion does the speaker most likely belong?
A. Hinduism
B. Buddhism
C. Shintoism
D. Zoroastrianism
Answer: A

This question refers to the following information.
"The struggle against neo-colonialism is not aimed at excluding the capital of the developed world from operating in less developed countries. It is aimed at preventing the financial power of the developed countries being used in such a way as to impoverish the less developed.
Non-alignment, as practiced by Ghana and many other countries, is based on co-operation with all States whether they be capitalist, socialist or have a mixed economy. Such a policy, therefore, involves foreign investment from capitalist countries, but it must be invested in accordance with a national plan drawn up by the government of the non-aligned State with its own interests in mind. The issue is not what return the foreign investor receives on his investments…The question is one of power. A State in the grip of neo-colonialism is not master of its own destiny."
Kwame Nkrumah, Neo-Colonialism, 1965
Which of the following provides the best context for Nkrumah's writings?
A. The Industrial Revolution
B. Decolonization
C. Regional Free Trade Associations
D. Autarky
Answer: B

This question refers to the following information.
"The real grievance of the worker is the insecurity of his existence; he is not sure that he will always have work, he is not sure that he will always be healthy, and he foresees that he will one day be old and unfit to work. If he falls into poverty, even if only through a prolonged illness, he is then completely helpless, exam_ins to his own devices, and society does not currently recognize any real obligation towards him beyond the usual help for the poor, even if he has been working all the time ever so faithfully and diligently. The usual help for the poor, however, leaves a lot to be desired, especially in large cities, where it is very much worse than in the country."
Otto von Bismarck, 1884
Otto von Bismarck likely made this speech in reaction to which of the following issues?
A. Social acceptance of child labor
B. Declining life expectancy in Germany
C. Criticisms of German trade tariffs
D. Negative effects attributed to industrial capitalism
Answer: D

This question refers to the following information.
"Indeed, as both the fatwas of distinguished [scholars] who base their opinion on reason and tradition alike and the consensus of the Sunni community agree that the ancient obligation of extirpation, extermination, and expulsion of evil innovation must be the aim of our exalted aspiration, for "Religious zeal is a victory for the Faith of God the Beneficent"; then, in accordance with the words of the Prophet (Peace upon him!) "Whosoever introduces evil innovation into our order must be expelled" and "Whosoever does aught against our order must be expelled," action has become necessary and exigent…"
Letter from Ottoman Sultan Selim I to Safavid Shah Ismail I, 1514
The letter from Selim I is most clearly an example of which of the following?
A. The maintenance of military supremacy at all costs
B. Expanding tensions between religious sects
C. Factors that brought about the collapse of the Ottoman Empire
D. Peacemaking efforts among the Islamic empires
Answer: B

This question refers to the following information.
"At least one of the [world's] societies would have to somehow enormously increase its productivity [in order to achieve global hegemony]. That quantum jump would have to be made before the various scientific, technological, agricultural, and industrial revolutions on which our post-quantum-leap world rests. It could only be accomplished by exploiting the ecosystems, mineral resources, and human assets of whole continents outside the lands of the society making the jump. Western Europe did just that by means of its brutality and guns and, more important, by geographical and ecological luck."
Copyright © 2015 Cambridge University Press.
Alfred Crosby, historian, Ecological Imperialism, 2004
The "quantum jump" mentioned in the passage most directly contributed to which of the following developments in the period 1450–1750 C.E.?
A. A breakdown in trade routes through the collapse of the established state structure
B. An increase in the population of the world through more plentiful supplies of food
C. The spread of Chinese and Indian belief systems across the world
D. An increase in social unrest
Answer: B

This question refers to the following information.
Is it not unity alone that can weld us into an effective force, capable of creating our own progress and making our valuable contribution to world peace? Which independent African state will claim that its financial structure and banking institutions are fully harnessed to its national development? Which will claim that its material resources and human energies are available for its own national aspirations? We are fast learning that political independence is not enough to rid us of the consequences of colonial rule. We have been too busy nursing our separate states to understand fully the basic need for union, rooted in common purpose, common planning and common endeavour.
Ghana's president, Kwame Nkrumah, addressing the Organization of African Unity, 1963
The speaker in the passage above is espousing which of the following causes?
A. Nationalism
B. Socialism
C. Pan-Africanism
D. Neocolonialism
Answer:
```
Answer: **C**

### 16. [prehistory]

```
The following are multiple choice questions (with answers) about prehistory.

The great Mayan king Pacal built temples in the city of Palenque in order to:
A. satisfy the powerful Mayan astronomer priests.
B. display his generosity to the common people, since they were allowed to live in the temples.
C. frighten away enemies, in particular the Spaniards.
D. legitimize his kingship, since his father was not royal.
Answer: D

According to Timothy Pauketat, the evidence for social stratification and political power at Cahokia suggests:
A. a center of Mississippian civilization with conditions similar to the rise of early states.
B. the limitations of authority in a Native American society of egalitarian foragers.
C. a simple chiefdom or perhaps a complex chiefdom had evolved by A.D. 1500.
D. a center of Mississippian civilization with conditions similar to societies on the Northwest Coast of North America.
Answer: A

Researchers now believe that the decline of the Maya was caused chiefly by:
A. a cataclysm of some kind, such as an earthquake, volcano, or tsunami.
B. ecological degradation resulting from slash-and-burn farming techniques.
C. endless wars between neighboring Mayan city-states.
D. practices of interbreeding that led to a steep rise in congenital disorders.
Answer: B

Recent research on hominid species dating from the Middle Pliocene indicates there was (as of 2020):
A. a great amount of species diversity, or a single species that exhibited a lot of diversity.
B. very little species diversity during this period and very few hominids.
C. decreased species diversity due to a prolonged ice age followed by a severe drought.
D. decreased species diversity but increased numbers of hammerstones and flakes, indicating stone tool manufacture.
Answer: A

What is the approximate mean cranial capacity of Homo erectus?
A. under 650 cc
B. about 800 cc
C. just under 1000 cc
D. 1200 cc
Answer: C

Monte Alban is located in the ________ and was built by the _________.
A. Valley of Mexico; Aztec.
B. Valley of Oaxaca; Zapotec.
C. Amazon floodplain; Olmec.
D. Yucatán Peninsula; Maya.
Answer:
```
Answer: **B**

### 17. [college_medicine]

```
The following are multiple choice questions (with answers) about college medicine.

Glucose is transported into the muscle cell:
A. via protein transporters called GLUT4.
B. only in the presence of insulin.
C. via hexokinase.
D. via monocarbylic acid transporters.
Answer: A

Which of the following is not a true statement?
A. Muscle glycogen is broken down enzymatically to glucose-1-phosphate
B. Elite endurance runners have a high proportion of Type I fibres in their leg muscles
C. Liver glycogen is important in the maintenance of the blood glucose concentration
D. Insulin promotes glucose uptake by all tissues in the body
Answer: D

In a genetic test of a newborn, a rare genetic disorder is found that has X-linked recessive transmission. Which of the following statements is likely true regarding the pedigree of this disorder?
A. All descendants on the maternal side will have the disorder.
B. Females will be approximately twice as affected as males in this family.
C. All daughters of an affected male will be affected.
D. There will be equal distribution of males and females affected.
Answer: C

A high school science teacher fills a 1 liter bottle with pure nitrogen and seals the lid. The pressure is 1.70 atm, and the room temperature is 25°C. Which two variables will both increase the pressure of the system, if all other variables are held constant?
A. Increasing temperature, increasing moles of gas
B. Increasing temperature, increasing volume
C. Decreasing volume, decreasing temperature
D. Decreasing moles of gas, increasing volume
Answer: A

An expected side effect of creatine supplementation is:
A. muscle weakness.
B. gain in body mass.
C. muscle cramps.
D. loss of electrolytes.
Answer: B

Sauna use, sometimes referred to as "sauna bathing," is characterized by short-term passive exposure to extreme heat. This exposure elicits mild hyperthermia – an increase in the body's core temperature – that induces a thermoregulatory response involving neuroendocrine, cardiovascular, and cytoprotective mechanisms that work together to restore homeostasis and condition the body for future heat stressors… In recent decades, sauna bathing has emerged as a means to increase lifespan and improve overall health, based on compelling data from observational, interventional, and mechanistic studies. Of particular interest are the findings from studies of participants in the Kuopio Ischemic Heart Disease Risk Factor (KIHD) Study, an ongoing prospective population-based cohort study of health outcomes in more than 2,300 middle-aged men from eastern Finland, which identified strong links between sauna use and reduced death and disease… The KIHD findings showed that men who used the sauna two to three times per week were 27 percent less likely to die from cardiovascular-related causes than men who didn't use the sauna.[2] Furthermore, the benefits they experienced were found to be dose-dependent: Men who used the sauna roughly twice as often, about four to seven times per week, experienced roughly twice the benefits – and were 50 percent less likely to die from cardiovascular-related causes.[2] In addition, frequent sauna users were found to be 40 percent less likely to die from all causes of premature death. These findings held true even when considering age, activity levels, and lifestyle factors that might have influenced the men's health.[2]... The KIHD also revealed that frequent sauna use reduced the risk of developing dementia and Alzheimer's disease in a dose-dependent manner. Men who used the sauna two to three times per week had a 66 percent lower risk of developing dementia and a 65 percent lower risk of developing Alzheimer's disease, compared to men who used the sauna only one time per week… The health benefits associated with sauna use extended to other aspects of mental health, as well. Men participating in the KIHD study who used the sauna four to seven times per week were 77 percent less likely to develop psychotic disorders, regardless of the men's dietary habits, socioeconomic status, physical activity, and inflammatory status (as measured by C-reactive protein)…Exposure to high temperature stresses the body, eliciting a rapid, robust response. The skin and core body temperatures increase markedly, and sweating ensues. The skin heats first, rising to 40°C (104°F), and then changes in core body temperature occur, rising slowly from 37°C (98.6°F, or normal) to 38°C (100.4°F) and then rapidly increasing to 39°C (102.2°F)…  Cardiac output, a measure of the amount of work the heart performs in response to the body's need for oxygen, increases by 60 to 70 percent, while the heart rate (the number of beats per minute) increases and the stroke volume (the amount of blood pumped) remains unchanged.[5] During this time, approximately 50 to 70 percent of the body's blood flow is redistributed from the core to the skin to facilitate sweating. The average person loses approximately 0.5 kg of sweat while sauna bathing.[11] Acute heat exposure also induces a transient increase in overall plasma volume to mitigate the decrease in core blood volume. This increase in plasma volume not only provides a reserve source of fluid for sweating, but it also acts like the water in a car's radiator, cooling the body to prevent rapid increases in core body temperature and promoting heat tolerance… Repeated sauna use acclimates the body to heat and optimizes the body's response to future exposures, likely due to a biological phenomenon known as hormesis, a compensatory defense response following exposure to a mild stressor that is disproportionate to the magnitude of the stressor. Hormesis triggers a vast array of protective mechanisms that not only repair cell damage but also provide protection from subsequent exposures to more devastating stressors… The physiological responses to sauna use are remarkably similar to those experienced during moderate- to vigorous-intensity exercise. In fact, sauna use has been proposed as an alternative to exercise for people who are unable to engage in physical activity due to chronic disease or physical limitations.[13]

Based on the article, which of the following statements is the author likely to agree with?
A. Heart surgery patients who cannot run on treadmills may benefit from sauna use.
B. Patients on a diet would benefit from sauna use.
C. Salt restriction would be equal to sauna use for hypertensive patients.
D. Patients with skin conditions may be cured with sauna use.
Answer:
```
Answer: **A**

### 18. [professional_law]

```
The following are multiple choice questions (with answers) about professional law.

A state legislature has recently enacted a statute making it a misdemeanor to curse or revile or use obscene or opprobrious language toward or in reference to a police officer perfonning his duties. A student at a state university organized a demonstration on campus to protest the war. The rally was attended by a group of 50 students who shouted anti-war messages at cars passing by. To show his contempt for the United States, the student sewed the American flag to the rear of his jeans. When a police officer saw the flag sown on the student's jeans, he approached and told him to remove the flag or he would be placed under arrest. The student became angered and shouted at the police officer, "Listen, you bastard, I'll wear this rag anywhere I please. " The student was subsequently placed under arrest and charged with violating the state statute. The student subsequently brings suit in state court challenging the constitutionality of the statute. The strongest constitutional argument for the student is that
A. the statute is void for vagueness under the Fourteenth Amendment's due process clause.
B. the statute is invalid because it violates the petitioner's freedom of speech under the First Amendment.
C. the statute is an abridgment of freedom of speech under the First Amendment because less restrictive means are available for achieving the same purpose.
D. the statute is overbroad and consequently invalid under the First and FourteenthAmendments.
Answer: D

A state has recently enacted a statute prohibiting the disposal of any nuclear wastes within the state. This law does not contravene or conflict with any federal statutes. A man operates a company in the state that is engaged in the disposal of nuclear wastes. Subsequent to the passage of the state statute, the man, not yet aware of the new law, entered into contracts with many out-of-state firms to dispose of their nuclear wastes in the state. On account of this new law, however, the man will be unable to perform these contracts. Assume that the man has standing to challenge this state law. Which of the following presents his strongest constitutional grounds to challenge the state law prohibiting the disposal of nuclear wastes within the state?
A. The commerce clause.
B. The equal protection clause of the Fourteenth Amendment.
C. The privileges and immunities clause of Article IV, Section 2. 
D. The contract clause.
Answer: A

Judge took judicial notice of some facts at the beginning of the trial. Which of the following is not an appropriate kind of fact for judicial notice?
A. Indisputable facts.
B. Facts that have been asserted by individual political organizations.
C. Facts recognized to be true by common knowledge.
D. Facts capable of scientific verification.
Answer: B

On October 1, 1980, a developer, owner of several hundred acres in a rural county, drafted a general development plan for the area. The duly recorded plan imposed elaborate limitations and restrictions upon the land in the plan, which was to be developed as a residential district. The restrictions were to extend to all persons acquiring any of the lots and to their heirs, assigns, and lessees. It was further provided that all subsequent owners would be charged with due notice of the restrictions. Among those restrictions in the general plan were the following:(22) A franchise right is created in a strip of land 10 feet in width along the rear of each lot for the use of public utility companies with right of ingress and egress. (23) No house or structure of any kind shall be built on the aforementioned strip of land running through the said blocks. In 2000, a retiree purchased one of the lots, built a house, and erected a fence in the rear of his property within the restricted area. In 2004, a teacher purchased a lot adjacent to the retiree's property and built a new house. Two years later, a librarian purchased the lot that adjoined the teacher's property. The three deeds to those properties each contained references to the deed book where the general plan was recorded. In 2008, the librarian began the construction of a seven-foot post-and-rail fence along the line dividing his lot with the teacher's, and along the center of the area subject to the franchise right. Although the teacher objected to its construction, the fence was completed. If the teacher seeks a mandatory injunction to compel removal of the librarian's fence, the court will most likely
A. grant relief, because the fence was in violation of the easement restriction. 
B. grant relief, because the encroachment of the fence violated the restriction in the original plan. 
C. deny relief, because the teacher failed to enforce the restriction against the retiree. 
D. deny relief, because the fence would not be construed as "a structure" within the terms of the restriction. 
Answer: B

A son owed a creditor $5,000. The son's father contacted the creditor and told him that he wanted to pay the son's debt. The father signed a document that stated the father would pay the son's debt at a rate of $500 a month for 10 months. The creditor made no written or oral commitment to forbear to sue the son to collect the $5,000 debt, and the father made no oral or written request for any such forbearance. For the next five months, the father made and the creditor accepted the $500 monthly payments as agreed. During that period, the creditor, in fact, did forbear to take any legal action against the son. However, the father then informed the creditor that he would make no further payments on the debt. Which of the following is the most persuasive argument that the father is liable to the creditor under the terms of their agreement?
A. The father's promise and the creditor's reliance thereon, if proved, gave rise to a valid claim by the creditor against the father based on the doctrine of promissory estoppel. 
B. Because it was foreseeable that the father's promise would induce the creditor to forbear taking any action against the son, such forbearance was, as a matter of law, a bargained-for consideration for the father's promise. 
C. The father's five payments to the creditor totaling $2,500 manifested a serious intent on the father's part to be contractually bound, and such manifestation is generally recognized as an effective substitute for consideration. 
D. By assuming the antecedent debt obligation that the son owed to the creditor, the father became a surety whose promise to the creditor was enforceable, since it was in writing and supported by adequate consideration. 
Answer: A

A department store had suffered a succession of thefts of merchandise over a period of months. From reports by employees and customers, the department store's manager concluded that the losses were due, wholly or in large part, to the depredations of a female shoplifter, aged 30 to 40 years, about 5 feet 5 inches in height, with red hair and green eyes, who usually wore a suit. This information was passed on to all of the employees of the department store. One day, a woman entered the store to purchase accessories. The woman was 5 feet 5 inches tall, 37 years old, red-haired and green-eyed, and dressed in a smartly tailored suit. She carried a large shopping bag. The manager noticed her as she picked up, examined, and put down a number of gloves and scarves. After a while, she looked about tentatively, and then started to walk out, swinging her bag. The manager intercepted her and, standing in front of her, politely asked the woman if she would accompany her to the store manager's office. When the woman asked for an explanation, the manager told her of the store's recent experience and the suspicion that the woman might be concealing pilfered merchandise in her bag. Flushing angrily, the woman replied, "Very well," and followed her to the office. Once there, the manager began to question the woman quite intensively. After the questioning, the manager then asked the woman's permission to inspect the contents of her shopping bag. At first, the woman curtly refused and announced that she had had enough of this nonsense. When she rose to go, the manager told her, "Listen, unless you let me look inside that bag, I'm going to call the police. " The woman replied, "Very well," and handed her the bag. The manager inspected the contents of the bag thoroughly but did not find any pilfered merchandise. She then gave the woman permission to leave. The total length of the woman's detention was 30 minutes. If the woman asserts a claim for false imprisonment against the department store, she will most likely
A. prevail, because she was detained for an unreasonably long period of time.
B. prevail, because the manager did not have reasonable grounds to suspect that the woman had committed a theft.
C. not prevail, because under the circumstances, the manager had reasonable grounds to suspect that the woman had committed a theft.
D. not prevail, because the manager gave the woman permission to leave after conducting her investigation.
Answer:
```
Answer: **A**

### 19. [high_school_geography]

```
The following are multiple choice questions (with answers) about high school geography.

The rate of natural increase of a population is found by subtracting the
A. crude death rate from the crude birth date.
B. crude birth rate from the crude death rate.
C. doubling time from the crude birth rate.
D. fertility rate from the crude death rate.
Answer: A

During the third stage of the demographic transition model, which of the following is true?
A. Birth rates increase and population growth rate is less rapid.
B. Birth rates decline and population growth rate is less rapid.
C. Birth rates increase and population growth rate increases.
D. Birth rates decrease and population growth rate increases.
Answer: B

Which of the following statements is NOT accurate regarding the services provided by local governments in the United States?
A. Duplication of efforts occurs often.
B. Social problems of the central city spill over into the surrounding residential suburbs.
C. Inefficiency in providing services occurs often.
D. One neighborhood's efforts to reduce pollution are always supported by neighboring communities.
Answer: D

The practice of hiring a foreign third-party service provider to run an operation is called
A. outsourcing.
B. offshoring.
C. maquiladoras.
D. locational interdependence.
Answer: B

Which one of the following items is an example of nonmaterial culture?
A. Dove soap
B. Dove candy bar
C. Dove symbol
D. A dove (bird)
Answer: C

Which language family is most widely spoken in North America and Europe and includes Baltic, Celtic, Germanic, and Greek?
A. Dravidian
B. Uralic-Altaic
C. Sino-Tibetan
D. Indo-European
Answer:
```
Answer: **D**

### 20. [moral_scenarios]

```
The following are multiple choice questions (with answers) about moral scenarios.

For which of these two scenarios does the main character (who uses I/me/my) do something clearly morally wrong, according to ordinary moral standards in the US as of 2020? Scenario 1 | I really loved my newborn baby boy. Scenario 2 | The new girl is embarrassed to live in a trailer park, so I invite the class to her house.
A. Wrong, Wrong
B. Wrong, Not wrong
C. Not wrong, Wrong
D. Not wrong, Not wrong
Answer: C

For which of these two scenarios does the main character (who uses I/me/my) do something clearly morally wrong, according to ordinary moral standards in the US as of 2020? Scenario 1 | I asked for a cigarette from the guy outside the store. Scenario 2 | When the security guard left his post, I pocketed a rare coin.
A. Wrong, Wrong
B. Wrong, Not wrong
C. Not wrong, Wrong
D. Not wrong, Not wrong
Answer: C

For which of these two scenarios does the main character (who uses I/me/my) do something clearly morally wrong, according to ordinary moral standards in the US as of 2020? Scenario 1 | I bought an old pick-up truck for work. Scenario 2 | I looked up at the gorgeous waterfall and threw a rock at it.
A. Wrong, Wrong
B. Wrong, Not wrong
C. Not wrong, Wrong
D. Not wrong, Not wrong
Answer: D

For which of these two scenarios does the main character (who uses I/me/my) do something clearly morally wrong, according to ordinary moral standards in the US as of 2020? Scenario 1 | During the house's demolition, I pushed him into a wall. Scenario 2 | When I choose a babysitter for my twin nieces, I don't care about her racial background.
A. Wrong, Wrong
B. Wrong, Not wrong
C. Not wrong, Wrong
D. Not wrong, Not wrong
Answer: B

For which of these two scenarios does the main character (who uses I/me/my) do something clearly morally wrong, according to ordinary moral standards in the US as of 2020? Scenario 1 | I took his wallet after he harassed me at the bar. Scenario 2 | I made love to my newborn baby boy.
A. Wrong, Wrong
B. Wrong, Not wrong
C. Not wrong, Wrong
D. Not wrong, Not wrong
Answer: A

For which of these two scenarios does the main character (who uses I/me/my) do something clearly morally wrong, according to ordinary moral standards in the US as of 2020? Scenario 1 | I decided to cook a potato for my girlfriend. Scenario 2 | I cornered the rat forcing it to go outside
A. Wrong, Wrong
B. Wrong, Not wrong
C. Not wrong, Wrong
D. Not wrong, Not wrong
Answer:
```
Answer: **D**

### 21. [virology]

```
The following are multiple choice questions (with answers) about virology.

Why are parvoviruses a highly impactful parasite?
A. Because they have no nucleic acid
B. They require a helper virus
C. Only replicate in dividing cells
D. Can integrate into host chromosomes
Answer: A

Which of the following is a morphological characteristic of the paramyxoviruses.
A. Fragile viruses often visualised with RNA spewing from the inside
B. Elongate viruses
C. Icosahedral viruses with envelope
D. Very large viruses
Answer: A

A key factor facilitating the application of nested case-control studies from the MACS was:
A. Data collection
B. Establishment of a repository of biologic specimens
C. Participant interest
D. Administration of the questionnaire by staff
Answer: B

The most important goal of a behavioral intervention is:
A. Change in behavior
B. Comprehensive coverage
C. Effective use of behavioral theory
D. Sustained behavior change
Answer: D

The median survival time to AIDS and death was established by following:
A. Seroprevalent HIV-infected individuals
B. Seronegatives
C. Seroconverters
D. High-risk seronegatives
Answer: C

How does rubella cause foetal abnormalities?
A. By crossing the placenta early in pregnancy and infecting the foetus
B. By only infecting the placenta
C. By inducing cytokines and chemokines in the mother
D. By raising the temperature of the mother and inducing an abnormal immune reaction to the foetus
Answer:
```
Answer: **A**

### 22. [elementary_mathematics]

```
The following are multiple choice questions (with answers) about elementary mathematics.

The population of the city where Michelle was born is 145,826. What is the value of the 5 in the number 145,826?
A. 5 thousands
B. 5 hundreds
C. 5 tens
D. 5 ones
Answer: A

Olivia used the rule "Add 11" to create the number pattern shown below. 10, 21, 32, 43, 54 Which statement about the number pattern is true?
A. The 10th number in the pattern will be an even number.
B. The number pattern will never have two even numbers next to each other.
C. The next two numbers in the pattern will be an even number then an odd number.
D. If the number pattern started with an odd number then the pattern would have only odd numbers in it.
Answer: B

A total of 30 players will play basketball at a park. There will be exactly 5 players on each team. Which statement correctly explains how to find the number of teams needed?
A. Add 5 to 30 to find 35 teams.
B. Divide 30 by 5 to find 6 teams.
C. Multiply 30 and 5 to find 150 teams.
D. Subtract 5 from 30 to find 25 teams.
Answer: B

A store sells 107 different colors of paint. They have 25 cans of each color in storage. The number of cans of paint the store has in storage can be found using the expression below. 107 × 25. How many cans of paint does the store have in storage?
A. 749
B. 2,675
C. 2,945
D. 4,250
Answer: B

Which expression is equivalent to 5 x 9?
A. (5 x 4) x (6 x 5)
B. (5 x 5) + (5 x 4)
C. (5 x 5) + (5 x 9)
D. (5 x 9) x (6 x 9)
Answer: B

Find the opposite of the number −7.
A. 7
B. −7
C. 0.7
D. −7.7
Answer:
```
Answer: **A**

### 23. [high_school_chemistry]

```
The following are multiple choice questions (with answers) about high school chemistry.

Which of the following is considered an acid anhydride?
A. HCl
B. H2SO3
C. SO2
D. Al(NO3)3
Answer: C

Which of the following is expected to be a polar molecule?
A. PCl4F
B. BF3
C. CO2
D. Si(CH3)4
Answer: A

From the solubility rules, which of the following is true?
A. All chlorides, bromides, and iodides are soluble
B. All sulfates are soluble
C. All hydroxides are soluble
D. All ammonium-containing compounds are soluble
Answer: D

A new compound is synthesized and found to be a monoprotic acid with a molar mass of 248 g/mol. When 0.0050 mol of this acid are dissolved in 0.500 L of water, the pH is measured as 3.89. What is the pKa of this acid?
A. 3.89
B. 7.78
C. 5.78
D. 2.33
Answer: C

A solution contains 2.00 mole of acetic acid, CH3COOH, and 1.00 mole of calcium acetate, Ca(CH3COO)2. The solution is able to resist the addition of a small amount of strong acid or strong base with only minor changes in the pH of the solution. Larger quantities of strong acid or strong base can cause a significant change in pH. How many moles of nitric acid, HNO3, may be added before the pH begins to change significantly?
A. 0.500 mole
B. 1.00 mole
C. 2.00 mole
D. 3.00 mole
Answer: C

Silver metal, often amalgamated with mercury, is used to reduce substances to a desired oxidation state. If the silver metal amalgam cannot be used because of concerns about mercury, which of the following would be a reasonable and safe substitute?
A. H+(aq)
B. Na(s)
C. Ca2+(aq)
D. Mg(s)
Answer:
```
Answer: **D**

### 24. [professional_medicine]

```
The following are multiple choice questions (with answers) about professional medicine.

A 42-year-old man comes to the office for preoperative evaluation prior to undergoing adrenalectomy scheduled in 2 weeks. One month ago, he received care in the emergency department for pain over his right flank following a motor vehicle collision. At that time, blood pressure was 160/100 mm Hg and CT scan of the abdomen showed an incidental 10-cm left adrenal mass. Results of laboratory studies, including complete blood count, serum electrolyte concentrations, and liver function tests, were within the reference ranges. The patient otherwise had been healthy and had never been told that he had elevated blood pressure. He takes no medications. A follow-up visit in the office 2 weeks ago disclosed elevated urinary normetanephrine and metanephrine and plasma aldosterone concentrations. The patient was referred to a surgeon, who recommended the adrenalectomy. Today, vital signs are temperature 36.6°C (97.9°F), pulse 100/min, respirations 14/min, and blood pressure 170/95 mm Hg. Physical examination discloses no significant findings. Initial preoperative preparation should include treatment with which of the following?
A. Labetalol
B. A loading dose of potassium chloride
C. Nifedipine
D. Phenoxybenzamine
Answer: D

A 36-year-old male presents to the office with a 3-week history of low back pain. He denies any recent trauma but says that he climbs in and out of his truck numerous times a day for his job. Examination of the patient in the prone position reveals a deep sacral sulcus on the left, a posterior inferior lateral angle on the right, and a lumbosacral junction that springs freely on compression. The most likely diagnosis is
A. left-on-left sacral torsion
B. left-on-right sacral torsion
C. right unilateral sacral flexion
D. right-on-right sacral torsion
Answer: D

A previously healthy 32-year-old woman comes to the physician 8 months after her husband was killed in a car crash. Since that time, she has had a decreased appetite and difficulty falling asleep. She states that she is often sad and cries frequently. She has been rechecking the door lock five times before leaving her house and has to count exactly five pieces of toilet paper before she uses it. She says that she has always been a perfectionist but these urges and rituals are new. Pharmacotherapy should be targeted to which of the following neurotransmitters?
A. Dopamine
B. Glutamate
C. Norepinephrine
D. Serotonin
Answer: D

A 44-year-old man comes to the office because of a 3-day history of sore throat, nonproductive cough, runny nose, and frontal headache. He says the headache is worse in the morning and ibuprofen does provide some relief. He has not had shortness of breath. Medical history is unremarkable. He takes no medications other than the ibuprofen for pain. Vital signs are temperature 37.4°C (99.4°F), pulse 88/min, respirations 18/min, and blood pressure 120/84 mm Hg. Examination of the nares shows erythematous mucous membranes. Examination of the throat shows erythema and follicular lymphoid hyperplasia on the posterior oropharynx. There is no palpable cervical adenopathy. Lungs are clear to auscultation. Which of the following is the most likely cause of this patient's symptoms?
A. Allergic rhinitis
B. Epstein-Barr virus
C. Mycoplasma pneumoniae
D. Rhinovirus
Answer: D

A 22-year-old male marathon runner presents to the office with the complaint of right-sided rib pain when he runs long distances. Physical examination reveals normal heart and lung findings and an exhalation dysfunction at ribs 4-5 on the right. Which of the following muscles or muscle groups will be most useful in correcting this dysfunction utilizing a direct method?
A. anterior scalene
B. latissimus dorsi
C. pectoralis minor
D. quadratus lumborum
Answer: C

A 52-year-old man is brought to the emergency department 30 minutes after he had an episode of chest pain radiating to his jaw while shoveling snow. His pulse is 80/min, and blood pressure is 130/70 mm Hg. The lungs are clear to auscultation. Cardiac examination shows an S4. While undergoing an ECG, the patient says that he feels the chest pain returning. The most appropriate immediate treatment is a drug with which of the following mechanisms of action?
A. Increases cAMP concentration
B. Increases nitric oxide concentration
C. Inhibits potassium flux
D. Inhibits sodium flux
Answer:
```
Answer: **B**

### 25. [moral_scenarios]

```
The following are multiple choice questions (with answers) about moral scenarios.

For which of these two scenarios does the main character (who uses I/me/my) do something clearly morally wrong, according to ordinary moral standards in the US as of 2020? Scenario 1 | I really loved my newborn baby boy. Scenario 2 | The new girl is embarrassed to live in a trailer park, so I invite the class to her house.
A. Wrong, Wrong
B. Wrong, Not wrong
C. Not wrong, Wrong
D. Not wrong, Not wrong
Answer: C

For which of these two scenarios does the main character (who uses I/me/my) do something clearly morally wrong, according to ordinary moral standards in the US as of 2020? Scenario 1 | I asked for a cigarette from the guy outside the store. Scenario 2 | When the security guard left his post, I pocketed a rare coin.
A. Wrong, Wrong
B. Wrong, Not wrong
C. Not wrong, Wrong
D. Not wrong, Not wrong
Answer: C

For which of these two scenarios does the main character (who uses I/me/my) do something clearly morally wrong, according to ordinary moral standards in the US as of 2020? Scenario 1 | I bought an old pick-up truck for work. Scenario 2 | I looked up at the gorgeous waterfall and threw a rock at it.
A. Wrong, Wrong
B. Wrong, Not wrong
C. Not wrong, Wrong
D. Not wrong, Not wrong
Answer: D

For which of these two scenarios does the main character (who uses I/me/my) do something clearly morally wrong, according to ordinary moral standards in the US as of 2020? Scenario 1 | During the house's demolition, I pushed him into a wall. Scenario 2 | When I choose a babysitter for my twin nieces, I don't care about her racial background.
A. Wrong, Wrong
B. Wrong, Not wrong
C. Not wrong, Wrong
D. Not wrong, Not wrong
Answer: B

For which of these two scenarios does the main character (who uses I/me/my) do something clearly morally wrong, according to ordinary moral standards in the US as of 2020? Scenario 1 | I took his wallet after he harassed me at the bar. Scenario 2 | I made love to my newborn baby boy.
A. Wrong, Wrong
B. Wrong, Not wrong
C. Not wrong, Wrong
D. Not wrong, Not wrong
Answer: A

For which of these two scenarios does the main character (who uses I/me/my) do something clearly morally wrong, according to ordinary moral standards in the US as of 2020? Scenario 1 | I voted twice in the presidential election. Scenario 2 | I rolled the grocery cart to my car.
A. Wrong, Wrong
B. Wrong, Not wrong
C. Not wrong, Wrong
D. Not wrong, Not wrong
Answer:
```
Answer: **B**

### 26. [high_school_chemistry]

```
The following are multiple choice questions (with answers) about high school chemistry.

Which of the following is considered an acid anhydride?
A. HCl
B. H2SO3
C. SO2
D. Al(NO3)3
Answer: C

Which of the following is expected to be a polar molecule?
A. PCl4F
B. BF3
C. CO2
D. Si(CH3)4
Answer: A

From the solubility rules, which of the following is true?
A. All chlorides, bromides, and iodides are soluble
B. All sulfates are soluble
C. All hydroxides are soluble
D. All ammonium-containing compounds are soluble
Answer: D

A new compound is synthesized and found to be a monoprotic acid with a molar mass of 248 g/mol. When 0.0050 mol of this acid are dissolved in 0.500 L of water, the pH is measured as 3.89. What is the pKa of this acid?
A. 3.89
B. 7.78
C. 5.78
D. 2.33
Answer: C

A solution contains 2.00 mole of acetic acid, CH3COOH, and 1.00 mole of calcium acetate, Ca(CH3COO)2. The solution is able to resist the addition of a small amount of strong acid or strong base with only minor changes in the pH of the solution. Larger quantities of strong acid or strong base can cause a significant change in pH. How many moles of nitric acid, HNO3, may be added before the pH begins to change significantly?
A. 0.500 mole
B. 1.00 mole
C. 2.00 mole
D. 3.00 mole
Answer: C

A mechanism is a sequence of elementary reactions that add up to the overall reaction stoichiometry. A substance that is produced in one elementary reaction and consumed in another is called
A. a catalyst
B. an intermediate
C. a reactant
D. a complex
Answer:
```
Answer: **B**

### 27. [professional_medicine]

```
The following are multiple choice questions (with answers) about professional medicine.

A 42-year-old man comes to the office for preoperative evaluation prior to undergoing adrenalectomy scheduled in 2 weeks. One month ago, he received care in the emergency department for pain over his right flank following a motor vehicle collision. At that time, blood pressure was 160/100 mm Hg and CT scan of the abdomen showed an incidental 10-cm left adrenal mass. Results of laboratory studies, including complete blood count, serum electrolyte concentrations, and liver function tests, were within the reference ranges. The patient otherwise had been healthy and had never been told that he had elevated blood pressure. He takes no medications. A follow-up visit in the office 2 weeks ago disclosed elevated urinary normetanephrine and metanephrine and plasma aldosterone concentrations. The patient was referred to a surgeon, who recommended the adrenalectomy. Today, vital signs are temperature 36.6°C (97.9°F), pulse 100/min, respirations 14/min, and blood pressure 170/95 mm Hg. Physical examination discloses no significant findings. Initial preoperative preparation should include treatment with which of the following?
A. Labetalol
B. A loading dose of potassium chloride
C. Nifedipine
D. Phenoxybenzamine
Answer: D

A 36-year-old male presents to the office with a 3-week history of low back pain. He denies any recent trauma but says that he climbs in and out of his truck numerous times a day for his job. Examination of the patient in the prone position reveals a deep sacral sulcus on the left, a posterior inferior lateral angle on the right, and a lumbosacral junction that springs freely on compression. The most likely diagnosis is
A. left-on-left sacral torsion
B. left-on-right sacral torsion
C. right unilateral sacral flexion
D. right-on-right sacral torsion
Answer: D

A previously healthy 32-year-old woman comes to the physician 8 months after her husband was killed in a car crash. Since that time, she has had a decreased appetite and difficulty falling asleep. She states that she is often sad and cries frequently. She has been rechecking the door lock five times before leaving her house and has to count exactly five pieces of toilet paper before she uses it. She says that she has always been a perfectionist but these urges and rituals are new. Pharmacotherapy should be targeted to which of the following neurotransmitters?
A. Dopamine
B. Glutamate
C. Norepinephrine
D. Serotonin
Answer: D

A 44-year-old man comes to the office because of a 3-day history of sore throat, nonproductive cough, runny nose, and frontal headache. He says the headache is worse in the morning and ibuprofen does provide some relief. He has not had shortness of breath. Medical history is unremarkable. He takes no medications other than the ibuprofen for pain. Vital signs are temperature 37.4°C (99.4°F), pulse 88/min, respirations 18/min, and blood pressure 120/84 mm Hg. Examination of the nares shows erythematous mucous membranes. Examination of the throat shows erythema and follicular lymphoid hyperplasia on the posterior oropharynx. There is no palpable cervical adenopathy. Lungs are clear to auscultation. Which of the following is the most likely cause of this patient's symptoms?
A. Allergic rhinitis
B. Epstein-Barr virus
C. Mycoplasma pneumoniae
D. Rhinovirus
Answer: D

A 22-year-old male marathon runner presents to the office with the complaint of right-sided rib pain when he runs long distances. Physical examination reveals normal heart and lung findings and an exhalation dysfunction at ribs 4-5 on the right. Which of the following muscles or muscle groups will be most useful in correcting this dysfunction utilizing a direct method?
A. anterior scalene
B. latissimus dorsi
C. pectoralis minor
D. quadratus lumborum
Answer: C

A 22-year-old woman comes to the office because of urticaria. This is her first episode of urticaria and it has occurred and then resolved several times in the past week. The history and physical examination disclose no abnormalities. Which of the following is the most appropriate course of action?
A. Determine the serum IgE concentration
B. Determine the total eosinophil count
C. Refer her to an allergist
D. Treat the symptoms
Answer:
```
Answer: **D**

### 28. [moral_scenarios]

```
The following are multiple choice questions (with answers) about moral scenarios.

For which of these two scenarios does the main character (who uses I/me/my) do something clearly morally wrong, according to ordinary moral standards in the US as of 2020? Scenario 1 | I really loved my newborn baby boy. Scenario 2 | The new girl is embarrassed to live in a trailer park, so I invite the class to her house.
A. Wrong, Wrong
B. Wrong, Not wrong
C. Not wrong, Wrong
D. Not wrong, Not wrong
Answer: C

For which of these two scenarios does the main character (who uses I/me/my) do something clearly morally wrong, according to ordinary moral standards in the US as of 2020? Scenario 1 | I asked for a cigarette from the guy outside the store. Scenario 2 | When the security guard left his post, I pocketed a rare coin.
A. Wrong, Wrong
B. Wrong, Not wrong
C. Not wrong, Wrong
D. Not wrong, Not wrong
Answer: C

For which of these two scenarios does the main character (who uses I/me/my) do something clearly morally wrong, according to ordinary moral standards in the US as of 2020? Scenario 1 | I bought an old pick-up truck for work. Scenario 2 | I looked up at the gorgeous waterfall and threw a rock at it.
A. Wrong, Wrong
B. Wrong, Not wrong
C. Not wrong, Wrong
D. Not wrong, Not wrong
Answer: D

For which of these two scenarios does the main character (who uses I/me/my) do something clearly morally wrong, according to ordinary moral standards in the US as of 2020? Scenario 1 | During the house's demolition, I pushed him into a wall. Scenario 2 | When I choose a babysitter for my twin nieces, I don't care about her racial background.
A. Wrong, Wrong
B. Wrong, Not wrong
C. Not wrong, Wrong
D. Not wrong, Not wrong
Answer: B

For which of these two scenarios does the main character (who uses I/me/my) do something clearly morally wrong, according to ordinary moral standards in the US as of 2020? Scenario 1 | I took his wallet after he harassed me at the bar. Scenario 2 | I made love to my newborn baby boy.
A. Wrong, Wrong
B. Wrong, Not wrong
C. Not wrong, Wrong
D. Not wrong, Not wrong
Answer: A

For which of these two scenarios does the main character (who uses I/me/my) do something clearly morally wrong, according to ordinary moral standards in the US as of 2020? Scenario 1 | I spent my afternoon swimming in the pool. Scenario 2 | I made my partner scream for help the entire evening.
A. Wrong, Wrong
B. Wrong, Not wrong
C. Not wrong, Wrong
D. Not wrong, Not wrong
Answer:
```
Answer: **C**

### 29. [conceptual_physics]

```
The following are multiple choice questions (with answers) about conceptual physics.

Compared with the mass of a uranium atom undergoing fission, the combined masses of the products after fission are
A. less
B. more
C. the same
D. zero
Answer: A

Things that are equivalent according to the equivalence principle are
A. space and time.
B. a traveling twin and a stay-at-home twin.
C. gravity and acceleration.
D. mass and energy.
Answer: C

Colors in a soap bubble result from light
A. converted to a different frequency
B. deflection
C. interference
D. polarization
Answer: C

A model airplane flies slower when flying into the wind and faster with wind at its back. When launched at right angles to the wind a cross wind its groundspeed compared with flying in still air is
A. the same
B. greater
C. less
D. either greater or less depending on wind speed
Answer: B

Which of these three elements has the most mass per nucleon?
A. Hydrogen
B. Iron
C. Uranium
D. Same in each
Answer: A

If you were to travel at a speed close to the speed of light, you could notice that your own
A. mass changes.
B. pulse decreases.
C. Both of these.
D. Neither of these.
Answer:
```
Answer: **D**

### 30. [professional_psychology]

```
The following are multiple choice questions (with answers) about professional psychology.

One of your therapy clients asks your advice about a good weight- reduction program. You have investigated the programs in the community and are enrolled in the one you consider the best. This program offers a $50 bonus to its patrons for each new person they bring into the program. Under these circumstances, your most appropriate response would be to
A. tell your client the pros and cons of each program you know about except for the one in which you are enrolled
B. recommend to your client the program in which you are enrolled and explain the $50 bonus you will receive
C. recommend to your client the program in which you are enrolled and offer to have the $50 bonus credited to your client's account in the program
D. tell your client the pros and cons of each program you know about, but do not claim the $50 bonus if your client enrolls in your program
Answer: D

There are three ways to measure the Central Tendency: the Mean, the Median and the Mode. From your knowledge about them, what is the mode?
A. less sensitive to extreme scores than the mean
B. more useful for skewed distributions
C. sensitive to extreme values and highly skewed distributions
D. the most frequently occurring number
Answer: D

In terms of Hofstede’s (1980) five cultural dimensions, the United States scores at the top of the scale on:
A. individualism.
B. individualism and power distance.
C. power distance and masculinity.
D. uncertainty avoidance.
Answer: A

Carl Jung believed that a client's transference:
A. is a fantasy that distracts the client from reality.
B. represents “mixed feelings” toward the therapist. 
C. "is a form of ""acting out."""
D. reflects the client’s personal and collective unconscious.
Answer: D

In the construction of a multiple regression equation for purposes of prediction, the optimal combination of measures is one in which the predictors
A. are uncorrelated with each other but are moderately correlated with the criterion
B. have low correlations with each other and low correlations with the criterion
C. are highly intercorrelated with each other and moderately correlated with the criterion
D. have low correlations with the criterion bur are moderately correlated with each other
Answer: A

Form A of a standardized personality test was given in the fall and again in the spring co the same group of people. The reliability estimate that resulted from this research is referred to as
A. external consistency
B. equivalence
C. stability
D. internal consistency
Answer:
```
Answer: **C**

### 31. [high_school_biology]

```
The following are multiple choice questions (with answers) about high school biology.

Which of the following is not a way to form recombinant DNA?
A. Translation
B. Conjugation
C. Specialized transduction
D. Transformation
Answer: A

A mutation in a bacterial enzyme changed a previously polar amino acid into a nonpolar amino acid. This amino acid was located at a site distant from the enzyme’s active site. How might this mutation alter the enzyme’s substrate specificity?
A. By changing the enzyme’s pH optimum
B. By changing the enzyme’s location in the cell
C. By changing the shape of the protein
D. An amino acid change away from the active site cannot alter the enzyme’s substrate specificity.
Answer: C

In animal cells, which of the following represents the most likely pathway that a secretory protein takes as it is synthesized in a cell?
A. Plasma membrane–Golgi apparatus–ribosome–secretory vesicle–rough ER
B. Ribosome–Golgi apparatus–rough ER–secretory vesicle–plasma membrane
C. Plasma membrane–Golgi apparatus–ribosome–secretory vesicle–rough ER
D. Ribosome–rough ER–Golgi apparatus–secretory vesicle–plasma membrane
Answer: D

Which of the following is not known to be involved in the control of cell division?
A. Cyclins
B. Protein kinases
C. Checkpoints
D. Fibroblast cells
Answer: D

Homologous structures are often cited as evidence for the process of natural selection. All of the following are examples of homologous structures EXCEPT
A. the wings of a bird and the wings of a bat
B. the flippers of a whale and the arms of a man
C. the pectoral fins of a porpoise and the flippers of a seal
D. the forelegs of an insect and the forelimbs of a dog
Answer: D

All of the following statements are true EXCEPT
A. thyroxine increases the rate of metabolism
B. insulin decreases storage of glycogen
C. vasopressin stimulates water reabsorption in the kidney
D. epinephrine increases blood sugar levels and heart rate
Answer:
```
Answer: **B**

### 32. [moral_scenarios]

```
The following are multiple choice questions (with answers) about moral scenarios.

For which of these two scenarios does the main character (who uses I/me/my) do something clearly morally wrong, according to ordinary moral standards in the US as of 2020? Scenario 1 | I really loved my newborn baby boy. Scenario 2 | The new girl is embarrassed to live in a trailer park, so I invite the class to her house.
A. Wrong, Wrong
B. Wrong, Not wrong
C. Not wrong, Wrong
D. Not wrong, Not wrong
Answer: C

For which of these two scenarios does the main character (who uses I/me/my) do something clearly morally wrong, according to ordinary moral standards in the US as of 2020? Scenario 1 | I asked for a cigarette from the guy outside the store. Scenario 2 | When the security guard left his post, I pocketed a rare coin.
A. Wrong, Wrong
B. Wrong, Not wrong
C. Not wrong, Wrong
D. Not wrong, Not wrong
Answer: C

For which of these two scenarios does the main character (who uses I/me/my) do something clearly morally wrong, according to ordinary moral standards in the US as of 2020? Scenario 1 | I bought an old pick-up truck for work. Scenario 2 | I looked up at the gorgeous waterfall and threw a rock at it.
A. Wrong, Wrong
B. Wrong, Not wrong
C. Not wrong, Wrong
D. Not wrong, Not wrong
Answer: D

For which of these two scenarios does the main character (who uses I/me/my) do something clearly morally wrong, according to ordinary moral standards in the US as of 2020? Scenario 1 | During the house's demolition, I pushed him into a wall. Scenario 2 | When I choose a babysitter for my twin nieces, I don't care about her racial background.
A. Wrong, Wrong
B. Wrong, Not wrong
C. Not wrong, Wrong
D. Not wrong, Not wrong
Answer: B

For which of these two scenarios does the main character (who uses I/me/my) do something clearly morally wrong, according to ordinary moral standards in the US as of 2020? Scenario 1 | I took his wallet after he harassed me at the bar. Scenario 2 | I made love to my newborn baby boy.
A. Wrong, Wrong
B. Wrong, Not wrong
C. Not wrong, Wrong
D. Not wrong, Not wrong
Answer: A

For which of these two scenarios does the main character (who uses I/me/my) do something clearly morally wrong, according to ordinary moral standards in the US as of 2020? Scenario 1 | I stopped by and gave my girlfriend some flowers before driving home to my wife. Scenario 2 | I copied and pasted so I could write my paper with sources found on the internet.
A. Wrong, Wrong
B. Wrong, Not wrong
C. Not wrong, Wrong
D. Not wrong, Not wrong
Answer:
```
Answer: **A**

### 33. [high_school_geography]

```
The following are multiple choice questions (with answers) about high school geography.

The rate of natural increase of a population is found by subtracting the
A. crude death rate from the crude birth date.
B. crude birth rate from the crude death rate.
C. doubling time from the crude birth rate.
D. fertility rate from the crude death rate.
Answer: A

During the third stage of the demographic transition model, which of the following is true?
A. Birth rates increase and population growth rate is less rapid.
B. Birth rates decline and population growth rate is less rapid.
C. Birth rates increase and population growth rate increases.
D. Birth rates decrease and population growth rate increases.
Answer: B

Which of the following statements is NOT accurate regarding the services provided by local governments in the United States?
A. Duplication of efforts occurs often.
B. Social problems of the central city spill over into the surrounding residential suburbs.
C. Inefficiency in providing services occurs often.
D. One neighborhood's efforts to reduce pollution are always supported by neighboring communities.
Answer: D

The practice of hiring a foreign third-party service provider to run an operation is called
A. outsourcing.
B. offshoring.
C. maquiladoras.
D. locational interdependence.
Answer: B

Which one of the following items is an example of nonmaterial culture?
A. Dove soap
B. Dove candy bar
C. Dove symbol
D. A dove (bird)
Answer: C

The majority of Kurds are found in which country?
A. Iran
B. Iraq
C. Turkey
D. Egypt
Answer:
```
Answer: **C**

### 34. [miscellaneous]

```
The following are multiple choice questions (with answers) about miscellaneous.

How many axles does a standard automobile have?
A. one
B. two
C. four
D. eight
Answer: B

What place is named in the title of the 1979 live album by rock legends Cheap Trick?
A. Budapest
B. Budokan
C. Bhutan
D. Britain
Answer: B

Who is the shortest man to ever win an NBA slam dunk competition?
A. Anthony 'Spud' Webb
B. Michael 'Air' Jordan
C. Tyrone 'Muggsy' Bogues
D. Julius 'Dr J' Erving
Answer: A

What is produced during photosynthesis?
A. hydrogen
B. nylon
C. oxygen
D. light
Answer: C

Which of these songs was a Top 10 hit for the rock band The Police?
A. 'Radio Ga-Ga'
B. 'Ob-la-di Ob-la-da'
C. 'De Do Do Do De Da Da Da'
D. 'In-a-Gadda-Da-Vida'
Answer: C

Including the bottom how many sides are on a square-based pyramid?
A. three
B. four
C. five
D. six
Answer:
```
Answer: **C**

### 35. [electrical_engineering]

```
The following are multiple choice questions (with answers) about electrical engineering.

In an SR latch built from NOR gates, which condition is not allowed
A. S=0, R=0
B. S=0, R=1
C. S=1, R=0
D. S=1, R=1
Answer: D

In a 2 pole lap winding dc machine , the resistance of one conductor is 2Ω and total number of conductors is 100. Find the total resistance
A. 200Ω
B. 100Ω
C. 50Ω
D. 10Ω
Answer: C

The coil of a moving coil meter has 100 turns, is 40 mm long and 30 mm wide. The control torque is 240*10-6 N-m on full scale. If magnetic flux density is 1Wb/m2 range of meter is
A. 1 mA.
B. 2 mA.
C. 3 mA.
D. 4 mA.
Answer: B

Two long parallel conductors carry 100 A. If the conductors are separated by 20 mm, the force per meter of length of each conductor will be
A. 100 N.
B. 0.1 N.
C. 1 N.
D. 0.01 N.
Answer: B

A point pole has a strength of 4π * 10^-4 weber. The force in newtons on a point pole of 4π * 1.5 * 10^-4 weber placed at a distance of 10 cm from it will be
A. 15 N.
B. 20 N.
C. 7.5 N.
D. 3.75 N.
Answer: A

One of the following is the primary function of an oscillator
A. produces sinusoidal oscillations
B. generates non sinusoidal waveforms
C. generates sustained oscillations at a constant amplitude and specific frequency
D. none of the above
Answer:
```
Answer: **C**

### 36. [security_studies]

```
The following are multiple choice questions (with answers) about security studies.

What distinguishes coercive diplomacy from military force?
A. Compellence is another term for coercive diplomacy, but covering a narrower set of criteria; compellence covers those threats aimed at initiating adversary action. A threat to coerce a state to give up part of its territory would count as coercive diplomacy, as long as that threat proactively initiates action before reactive diplomacy is taken.
B. Coercive diplomacy constitutes the threats of limited force to induce adversary's incentive to comply with the coercer's demands. It is an influence strategy that is intended to obtain compliance: the use of force to defeat an opponent first does not count. It leaves an element of choice with the target to comply, or to continue.
C. Military force, or the threat of military force, utilises fear to achieve strategic objectives. Coercive diplomacy is differentiated from this approach, because it does not use fear as a tool for coercing an adversary.
D. Coercive diplomacy is employed to use force but to limit its effects on the international community. Coercive diplomacy is an aggressive strategy that is intended to obtain compliance through defeat. It does not leave an element of choice with the target, the target either being forced to comply or engage in conflict. It seeks to control by imposing compliance by removing any opportunity for negotiation or concession.
Answer: B

Which of the following is the best lens through which to investigate the role of child soldiers?
A. Child soldiers are victims of combat that need re-education and rehabilitation.
B. Children and their mothers are not active subjects in warfare and are best considered as subjects in the private sphere.
C. Children are most often innocent bystanders in war and are best used as signifiers of peace.
D. Children have political subjecthood that is missed when they are considered as passive victims of warfare.
Answer: D

In order to become securitized, a threat must be presented in which of these ways?
A. As an existential threat that requires immediate and extraordinary action, posing a threat to the survival of the state or to societal security.
B. As requiring immediate and extraordinary action by the state, threatening the survival of a referent object and therefore warranting the use of measures not normally employed in the political realm.
C. As an urgent threat to the survival of the referent object, so serious that it legitimises the employment of extraordinary action in response.
D. As an urgent threat to the survival of the audience that requires extraordinary or emergency measures.
Answer: C

How can we best describe the relationship between the state-centric approach and the concept of human security?
A. There are such wide divisions within the human security framework regarding the nature of threats and referent objects that no widely applicable comparisons between state-centric approaches and human security can be drawn.
B. By adopting the framework of human security, the limitations of the realist state-centric approach become evident. Whilst human security defines the referent object as the person or population, state-centric approaches prioritise the security of the state, de-prioritizing the pursuit of human security.
C. The state-centric approach to security is a faction of human security, usually defined within the broad school of human security. By being state-centric this approach prioritises the individual as the referent object in security studies.
D. Both the state-centric and human-centric approaches to security are mutually exclusive and offer a sufficient analytic framework with which to understand the international security system. It is therefore the role of security analysts to determine which of these substantial concepts is correct, and which should be discarded.
Answer: B

What are the frameworks of analysis within which terrorism has been considered (as of 2020)?
A. Competition between larger nations has resulted in some countries actively supporting terrorist groups to undermine the strength of rival states. Terrorist networks are extended patronage clubs maintained and paid for by their donor states and are conceptualised as being like state actors, to be dealt with using military force.
B. Globalization has enabled the internationalization of terrorist activities by opening up their operational space, although coordination is still managed from a geographical base. This suggests that terrorist groups are nationally structured which means that terrorism cannot be considered in terms of a war to be defeated militarily without having serious implications on the indigenous population.
C. Terrorism can be viewed as a problem to be resolved by military means (war on terrorism), by normal police techniques (terrorism as crime), or as a medical problem with underlying causes and symptoms (terrorism as disease).
D. Terrorism is viewed as a criminal problem. The criminalization of terrorism has two important implications. Firstly, it suggests that terrorism can be eradicated - terrorists can be caught and brought to trial by normal judicial proceedings thereby removing the threat from society - and secondly, it suggests that preventative crime techniques are applicable to prevent its development.
Answer: C

In what ways, if any, can the environment be considered a security concern?
A. Environmental security entails a consideration of the security of the global environment, as well as its nested sub-systems and social systems beyond the boundaries of the nation state.
B. Environmental security is a critical security project in that it questions who and what is to be secured and from what threat by orthodox security policies, or whether linkages between environmental, security and development issues can be made.
C. Environmental security is a practical endeavour to assess how environmental change causes violent conflict within and between countries, and the ways in which environmental security can undermine national security.
D. All of these options. The environment is both an object to be secured and a source of risk, although it may mean different things to different people. Whilst deepening the concept of security it has both critical and practical dimensions although the utility of the concept is contested.
Answer:
```
Answer: **D**

### 37. [business_ethics]

```
The following are multiple choice questions (with answers) about business ethics.

Beyond the business case for engaging in CSR there are a number of moral arguments relating to: negative _______, the _______that corporations possess and the ________ of business and society.
A. Externalities, Power, Independence
B. Publicity, Insubstantial resources, Mutual dependence
C. Publicity, Power, Independence
D. Externalities, Power, Mutual dependence
Answer: D

_______ is the direct attempt to formally or informally manage ethical issues or problems, through specific policies, practices and programmes.
A. Corporate social responsibility
B. Business ethics management
C. Sustainability
D. Environmental management
Answer: B

To ensure the independence of the non-executive board members, they are a number of steps which can be taken, which include non-executives being drawn from _______ the company, being appointed for a _________ time period as well as being appointed _________.
A. Outside, Limited, Independently
B. Inside, Limited, Intermittently
C. Outside, Unlimited, Intermittently
D. Inside, Unlimited, Independently
Answer: A

Three contrasting tactics that CSO's can engage in to meet their aims are ________ which typically involves research and communication, ________, which may involve physically attacking a company's operations or ________, often involving some form of _______.
A. Non-violent direct action, Violent direct action, Indirect action, Boycott
B. Indirect action, Instrumental action, Non-violent direct action, Information campaign
C. Indirect action, Violent direct action, Non-violent direct-action Boycott
D. Non-violent direct action, Instrumental action, Indirect action, Information campaign
Answer: C

In contrast to _______, _______ aim to reward favourable behaviour by companies. The success of such campaigns have been heightened through the use of ___________, which allow campaigns to facilitate the company in achieving _________ .
A. Buycotts, Boycotts, Blockchain technology, Charitable donations
B. Buycotts, Boycotts, Digital technology, Increased Sales
C. Boycotts, Buyalls, Blockchain technology, Charitable donations
D. Boycotts, Buycotts, Digital technology, Increased Sales
Answer: D

________ are a business ethics management tool that have grown over the last few decades. Areas which these typically cover are compliance, corruption and environmental issues. However, they are constantly evolving, with for example, the onus on __________ having grown in recent years.
A. Annual reports, Company financial performance
B. Annual report, Internet - social media and networking
C. Codes of ethics, Company financial performance
D. Codes of ethics, Internet - social media and networking
Answer:
```
Answer: **D**

### 38. [conceptual_physics]

```
The following are multiple choice questions (with answers) about conceptual physics.

Compared with the mass of a uranium atom undergoing fission, the combined masses of the products after fission are
A. less
B. more
C. the same
D. zero
Answer: A

Things that are equivalent according to the equivalence principle are
A. space and time.
B. a traveling twin and a stay-at-home twin.
C. gravity and acceleration.
D. mass and energy.
Answer: C

Colors in a soap bubble result from light
A. converted to a different frequency
B. deflection
C. interference
D. polarization
Answer: C

A model airplane flies slower when flying into the wind and faster with wind at its back. When launched at right angles to the wind a cross wind its groundspeed compared with flying in still air is
A. the same
B. greater
C. less
D. either greater or less depending on wind speed
Answer: B

Which of these three elements has the most mass per nucleon?
A. Hydrogen
B. Iron
C. Uranium
D. Same in each
Answer: A

If an iron nucleus split in two, its fission fragments would have
A. less mass per nucleon
B. more mass per nucleon
C. the same mass per nucleon
D. either more or less mass per nucleon
Answer:
```
Answer: **B**

### 39. [high_school_statistics]

```
The following are multiple choice questions (with answers) about high school statistics.

Which of the following is a correct statement about correlation?
A. If the slope of the regression line is exactly 1, then the correlation is exactly 1.
B. If the correlation is 0, then the slope of the regression line is undefined.
C. Switching which variable is called x and which is called y changes the sign of the correlation.
D. The correlation r is equal to the slope of the regression line when z-scores for the y-variable are plotted against z-scores for the x-variable.
Answer: D

Suppose X and Y are random variables with E(X) = 37, var(X) = 5, E(Y) = 62, and var(Y) = 12. What are the expected value and variance of the random variable X + Y?
A. E(X + Y) = 99, var(X + Y) = 8.5
B. E(X + Y) = 99, var(X + Y) = 13
C. E(X + Y) = 99, var(X + Y) = 17
D. There is insufficient information to answer this question.
Answer: D

After a frost warning was issued, the owner of a large orange grove asked his workers to spray all his trees with water. The water was supposed to freeze and form a protective covering of ice around the orange blossom. Nevertheless, the owner suspected that some trees suffered considerable damage due to the frost. To estimate the proportion of trees that suffered more than 50 percent damage due to the frost, he took a random sample of 100 trees from his grove. What is the response variable in this experiment?
A. The proportion of trees that suffered more than 50 percent damage due to frost.
B. The number of trees affected by the frost.
C. The number of trees sampled from the grove.
D. For each sampled tree, whether it suffered more than 50 percent damage or at most 50 percent damage.
Answer: D

A new smartwatch is manufactured in one part of a factory, then secured for shipping in another, independent part of the factory. The weight of the smartwatch has a mean of 62 grams and a standard deviation of 1.0 grams. The weight of the packaging (box, user's guide, bubble wrap, etc.) has a mean of 456 grams and a standard deviation of 6 grams. Together, the distribution of the weight of the smartwatch and its packaging would have the following mean and standard deviation:
A. Mean 518 grams; standard deviation 7.0 grams
B. Mean 518 grams; standard deviation 3.5 grams
C. Mean 518 grams; standard deviation 6.1 grams
D. Mean 394 grams; standard deviation 6.1 grams
Answer: C

Which of the following sets has the smallest standard deviation? Which has the largest?
I: {1,2,3}
II: {-10,10}
III: {100}
A. I, II
B. II, III
C. III, I
D. III, II
Answer: D

In one metropolitan region, technical writers have an annual salary that is approximately normally distributed with a mean of $55,800. The first quartile of salaries is $48,815. What is the standard deviation?
A. $6,984 
B. $10,356 
C. $10,476 
D. $13,968 
Answer:
```
Answer: **B**

### 40. [world_religions]

```
The following are multiple choice questions (with answers) about world religions.

What is the sign of the covenant for Jewish males?
A. The rainbow
B. Circumcision
C. A son
D. Bar mitzvah
Answer: B

What is the Second Gem in Buddhism?
A. The Dharma
B. The Sangha
C. The Buddha
D. The Bodhisattva
Answer: A

In which dynasty was the "Mandate of Heaven" developed to legitimatize the new rulers?
A. Shang
B. Zhou
C. Han
D. Xia
Answer: B

Which Japanese government promoted a kind of national cult based on the emperor and his associations with kami?
A. Honen
B. Tanaka
C. Tokugawa
D. Meiji
Answer: D

How can the Upanishads be characterized?
A. Ritual texts
B. Philosophical texts
C. Hymns
D. Origin stories
Answer: B

What is the communal meal offered at the place of worship called in Sikhism?
A. Sangat
B. Langar
C. Gurdwara
D. Panth
Answer:
```
Answer: **B**

### 41. [moral_scenarios]

```
The following are multiple choice questions (with answers) about moral scenarios.

For which of these two scenarios does the main character (who uses I/me/my) do something clearly morally wrong, according to ordinary moral standards in the US as of 2020? Scenario 1 | I really loved my newborn baby boy. Scenario 2 | The new girl is embarrassed to live in a trailer park, so I invite the class to her house.
A. Wrong, Wrong
B. Wrong, Not wrong
C. Not wrong, Wrong
D. Not wrong, Not wrong
Answer: C

For which of these two scenarios does the main character (who uses I/me/my) do something clearly morally wrong, according to ordinary moral standards in the US as of 2020? Scenario 1 | I asked for a cigarette from the guy outside the store. Scenario 2 | When the security guard left his post, I pocketed a rare coin.
A. Wrong, Wrong
B. Wrong, Not wrong
C. Not wrong, Wrong
D. Not wrong, Not wrong
Answer: C

For which of these two scenarios does the main character (who uses I/me/my) do something clearly morally wrong, according to ordinary moral standards in the US as of 2020? Scenario 1 | I bought an old pick-up truck for work. Scenario 2 | I looked up at the gorgeous waterfall and threw a rock at it.
A. Wrong, Wrong
B. Wrong, Not wrong
C. Not wrong, Wrong
D. Not wrong, Not wrong
Answer: D

For which of these two scenarios does the main character (who uses I/me/my) do something clearly morally wrong, according to ordinary moral standards in the US as of 2020? Scenario 1 | During the house's demolition, I pushed him into a wall. Scenario 2 | When I choose a babysitter for my twin nieces, I don't care about her racial background.
A. Wrong, Wrong
B. Wrong, Not wrong
C. Not wrong, Wrong
D. Not wrong, Not wrong
Answer: B

For which of these two scenarios does the main character (who uses I/me/my) do something clearly morally wrong, according to ordinary moral standards in the US as of 2020? Scenario 1 | I took his wallet after he harassed me at the bar. Scenario 2 | I made love to my newborn baby boy.
A. Wrong, Wrong
B. Wrong, Not wrong
C. Not wrong, Wrong
D. Not wrong, Not wrong
Answer: A

For which of these two scenarios does the main character (who uses I/me/my) do something clearly morally wrong, according to ordinary moral standards in the US as of 2020? Scenario 1 | I knew my overweight teacher was dieting, so I offered her a big sundae. Scenario 2 | I found a wallet in the park and became its new owner.
A. Wrong, Wrong
B. Wrong, Not wrong
C. Not wrong, Wrong
D. Not wrong, Not wrong
Answer:
```
Answer: **A**

### 42. [professional_law]

```
The following are multiple choice questions (with answers) about professional law.

A state legislature has recently enacted a statute making it a misdemeanor to curse or revile or use obscene or opprobrious language toward or in reference to a police officer perfonning his duties. A student at a state university organized a demonstration on campus to protest the war. The rally was attended by a group of 50 students who shouted anti-war messages at cars passing by. To show his contempt for the United States, the student sewed the American flag to the rear of his jeans. When a police officer saw the flag sown on the student's jeans, he approached and told him to remove the flag or he would be placed under arrest. The student became angered and shouted at the police officer, "Listen, you bastard, I'll wear this rag anywhere I please. " The student was subsequently placed under arrest and charged with violating the state statute. The student subsequently brings suit in state court challenging the constitutionality of the statute. The strongest constitutional argument for the student is that
A. the statute is void for vagueness under the Fourteenth Amendment's due process clause.
B. the statute is invalid because it violates the petitioner's freedom of speech under the First Amendment.
C. the statute is an abridgment of freedom of speech under the First Amendment because less restrictive means are available for achieving the same purpose.
D. the statute is overbroad and consequently invalid under the First and FourteenthAmendments.
Answer: D

A state has recently enacted a statute prohibiting the disposal of any nuclear wastes within the state. This law does not contravene or conflict with any federal statutes. A man operates a company in the state that is engaged in the disposal of nuclear wastes. Subsequent to the passage of the state statute, the man, not yet aware of the new law, entered into contracts with many out-of-state firms to dispose of their nuclear wastes in the state. On account of this new law, however, the man will be unable to perform these contracts. Assume that the man has standing to challenge this state law. Which of the following presents his strongest constitutional grounds to challenge the state law prohibiting the disposal of nuclear wastes within the state?
A. The commerce clause.
B. The equal protection clause of the Fourteenth Amendment.
C. The privileges and immunities clause of Article IV, Section 2. 
D. The contract clause.
Answer: A

Judge took judicial notice of some facts at the beginning of the trial. Which of the following is not an appropriate kind of fact for judicial notice?
A. Indisputable facts.
B. Facts that have been asserted by individual political organizations.
C. Facts recognized to be true by common knowledge.
D. Facts capable of scientific verification.
Answer: B

On October 1, 1980, a developer, owner of several hundred acres in a rural county, drafted a general development plan for the area. The duly recorded plan imposed elaborate limitations and restrictions upon the land in the plan, which was to be developed as a residential district. The restrictions were to extend to all persons acquiring any of the lots and to their heirs, assigns, and lessees. It was further provided that all subsequent owners would be charged with due notice of the restrictions. Among those restrictions in the general plan were the following:(22) A franchise right is created in a strip of land 10 feet in width along the rear of each lot for the use of public utility companies with right of ingress and egress. (23) No house or structure of any kind shall be built on the aforementioned strip of land running through the said blocks. In 2000, a retiree purchased one of the lots, built a house, and erected a fence in the rear of his property within the restricted area. In 2004, a teacher purchased a lot adjacent to the retiree's property and built a new house. Two years later, a librarian purchased the lot that adjoined the teacher's property. The three deeds to those properties each contained references to the deed book where the general plan was recorded. In 2008, the librarian began the construction of a seven-foot post-and-rail fence along the line dividing his lot with the teacher's, and along the center of the area subject to the franchise right. Although the teacher objected to its construction, the fence was completed. If the teacher seeks a mandatory injunction to compel removal of the librarian's fence, the court will most likely
A. grant relief, because the fence was in violation of the easement restriction. 
B. grant relief, because the encroachment of the fence violated the restriction in the original plan. 
C. deny relief, because the teacher failed to enforce the restriction against the retiree. 
D. deny relief, because the fence would not be construed as "a structure" within the terms of the restriction. 
Answer: B

A son owed a creditor $5,000. The son's father contacted the creditor and told him that he wanted to pay the son's debt. The father signed a document that stated the father would pay the son's debt at a rate of $500 a month for 10 months. The creditor made no written or oral commitment to forbear to sue the son to collect the $5,000 debt, and the father made no oral or written request for any such forbearance. For the next five months, the father made and the creditor accepted the $500 monthly payments as agreed. During that period, the creditor, in fact, did forbear to take any legal action against the son. However, the father then informed the creditor that he would make no further payments on the debt. Which of the following is the most persuasive argument that the father is liable to the creditor under the terms of their agreement?
A. The father's promise and the creditor's reliance thereon, if proved, gave rise to a valid claim by the creditor against the father based on the doctrine of promissory estoppel. 
B. Because it was foreseeable that the father's promise would induce the creditor to forbear taking any action against the son, such forbearance was, as a matter of law, a bargained-for consideration for the father's promise. 
C. The father's five payments to the creditor totaling $2,500 manifested a serious intent on the father's part to be contractually bound, and such manifestation is generally recognized as an effective substitute for consideration. 
D. By assuming the antecedent debt obligation that the son owed to the creditor, the father became a surety whose promise to the creditor was enforceable, since it was in writing and supported by adequate consideration. 
Answer: A

A state law imposes penalties for "any public statement containing false or misleading information about a service or product." An airline falsely claimed in an advertisement that its competitor had an inferior safety record. The claim was based on erroneous information, found on the website of a nonprofit consumer advocacy group, that the airline assumed to be true. The airline was charged under the state law for making a false statement. No federal statute applies. Which of the following best supports the airline in a defense based on the First Amendment?
A. Its statement about the safety record was made without malice.
B. Its statement about the safety record was protected noncommercial speech.
C. The state law is a prior restraint.
D. The state law is overbroad.
Answer:
```
Answer: **D**

### 43. [sociology]

```
The following are multiple choice questions (with answers) about sociology.

Which of the following did the post-war welfare state of 1948 not aim to provide:
A. free health care and education for all
B. a minimum wage
C. full employment
D. universal welfare
Answer: B

What does Berger (1963) describe as a metaphor for social reality?
A. a fairground ride
B. a circus
C. a puppet theatre
D. a ballet
Answer: C

The shift from 'civil religion' to 'common religion' means that:
A. the increasing bureaucracy of the state has made religion only a marginal part of our lives
B. despite the weakening of traditional authority, our everyday lives and 'common sense' remain shaped by religious beliefs and values
C. religious participation in collective worship may have declined, but people still practise their faiths in private
D. people are much more likely to discuss their religious beliefs in public, informal settings
Answer: B

The term 'hegemony' refers to:
A. the tendency for the working class not to realize their own interests
B. a dominant ideology that legitimates economic, political and cultural power
C. a form of dual consciousness based on ideology and everyday experiences
D. a mode of payment given for outstanding topiary
Answer: B

Which of the following is not a problem associated with official statistics on strike action?
A. most strikes go unnoticed by employers and the mass media
B. not all industrial disputes will be reported by the employer
C. the definition of strikes excludes those that involve fewer than ten workers or last less than one day
D. it is hard to compare strikes that were measured in different ways
Answer: A

When Berger & Luckmann said that reality is socially constructed, they meant:
A. scientists are guided in their work by social values and interests, so they define and measure phenomena that will support their theories
B. people negotiate shared definitions of their situation and live according to these, often forgetting that these social worlds are not fixed and external
C. sociologists decide what constitutes social reality and measure only that
D. terms like 'reality' have no deeper meaning beyond the level of discourse
Answer:
```
Answer: **B**

### 44. [college_mathematics]

```
The following are multiple choice questions (with answers) about college mathematics.

Let V be the set of all real polynomials p(x). Let transformations T, S be defined on V by T:p(x) -> xp(x) and S:p(x) -> p'(x) = d/dx p(x), and interpret (ST)(p(x)) as S(T(p(x))). Which of the following is true?
A. ST = 0
B. ST = T
C. ST = TS
D. ST - TS is the identity map of V onto itself.
Answer: D

A tank initially contains a salt solution of 3 grams of salt dissolved in 100 liters of water. A salt solution containing 0.02 grams of salt per liter of water is sprayed into the tank at a rate of 4 liters per minute. The sprayed solution is continually mixed with the salt solution in the tank, and the mixture flows out of the tank at a rate of 4 liters per minute. If the mixing is instantaneous, how many grams of salt are in the tank after 100 minutes have elapsed?
A. 2
B. 2 - e^-2
C. 2 + e^-2
D. 2 + e^-4
Answer: D

Let A be a real 2x2 matrix. Which of the following statements must be true?
I. All of the entries of A^2 are nonnegative.
II. The determinant of A^2 is nonnegative.
III. If A has two distinct eigenvalues, then A^2 has two distinct eigenvalues.
A. I only
B. II only
C. III only
D. II and III only
Answer: B

Suppose that f(1 + x) = f(x) for all real x. If f is a polynomial and f(5) = 11, then f(15/2)
A. -11
B. 0
C. 11
D. 33/2
Answer: C

Let A be the set of all ordered pairs of integers (m, n) such that 7m + 12n = 22. What is the greatest negative number in the set B = {m + n : (m, n) \in A}?
A. -5
B. -4
C. -3
D. -2
Answer: B

If c > 0 and f(x) = e^x - cx for all real numbers x, then the minimum value of f is
A. f(c)
B. f(e^c)
C. f(1/c)
D. f(log c)
Answer:
```
Answer: **D**

### 45. [nutrition]

```
The following are multiple choice questions (with answers) about nutrition.

In a cohort study, the risk ratio of developing diabetes was 0.86 when comparing consumers of tea (the exposed) to those who did not drink tea (the unexposed). Which one statement is correct (according to knowledge in 2020)?
A. The tea drinkers have lower risk of developing diabetes.
B. The tea drinkers have higher risk of developing diabetes.
C. Based on the information given we cannot tell if the observed difference in disease risk is the result of chance.
D. The risk ratio is close to the value one, so there is no difference in disease risk between the two groups.
Answer: C

Which of the following statements is correct (according to knowledge in 2020)?
A. Consumers with phenylketonuria must avoid the consumption of the sweetener aspartame
B. Consumers with phenylketonuria must avoid the consumption of the sweetener saccharin
C. Consumers with phenylketonuria must avoid the consumption of the sweetener sucralose
D. Consumers with phenylketonuria must avoid the consumption of the sweetener acesulfame K
Answer: A

Which of the following is the most plausible explanation for the protective effect of dietary fibre against cancer of the colon, as of 2020?
A. Propionic acid, formed during colonic fibre fermentation inhibits liver fatty acid synthesis
B. Butyric acid, formed during colonic fibre fermentation stimulates "silencing" of the SLC5A8 tumour suppressor gene
C. None of these options are correct
D. Butyric acid, formed during colonic fibre fermentation stimulates anti-oxidant defences in the colon
Answer: D

Which of the following statements about iodine is correct, as of 2020?
A. 50% of adults consume iodine at levels below the RNI
B. Dairy products are a poor source of iodine
C. The iodine content of organic milk is generally lower that the level in non-organic milk
D. UK dietary reference values recommend an increase in iodine intake in pregnancy
Answer: C

What is the first-line drug for patients with type 2 diabetes and obesity, as of 2020?
A. Acarbose
B. Metformin
C. Sulphonylureas
D. Insulin
Answer: B

These factors increase risk of osteoporotic fracture:
A. High bone mineral density
B. High body weight
C. High lean mass
D. Poor muscle strength
Answer:
```
Answer: **D**

### 46. [college_biology]

```
The following are multiple choice questions (with answers) about college biology.

Which of the following represents an accurate statement concerning arthropods?
A. They possess an exoskeleton composed primarily of peptidoglycan.
B. They possess an open circulatory system with a dorsal heart.
C. They are members of a biologically unsuccessful phylum incapable of exploiting diverse habitats and nutrition sources.
D. They lack paired, jointed appendages.
Answer: B

In a given population, 1 out of every 400 people has a cancer caused by a completely recessive allele, b. Assuming the population is in Hardy-Weinberg equilibrium, which of the following is the expected proportion of individuals who carry the b allele but are not expected to develop the cancer?
A. 1/400
B. 19/400
C. 20/400
D. 38/400
Answer: D

The presence of homologous structures in two different organisms, such as the humerus in the front limb of a human and a bird, indicates that
A. the human and bird are polyphyletic species
B. a human's and bird's evolution is convergent
C. the human and bird belong to a clade
D. the human and bird developed by analogy
Answer: C

According to the pressure-flow model of movement of phloem contents, photosynthate movement from source to sink is driven by
A. an ATP-dependent pressure-flow pump
B. a water-pressure potential gradient
C. transpiration
D. apoplastic diffusion
Answer: B

Which of the following contain DNA sequences required for the segregation of chromosomes in mitosis and meiosis?
A. Telomeres
B. Centromeres
C. Nucleosomes
D. Spliceosomes
Answer: B

To prevent desiccation and injury, the embryos of terrestrial vertebrates are encased within a fluid secreted by the
A. amnion
B. chorion
C. allantois
D. yolk sac
Answer:
```
Answer: **A**

### 47. [college_computer_science]

```
The following are multiple choice questions (with answers) about college computer science.

Which of the following regular expressions is equivalent to (describes the same set of strings as) (a* + b)*(c + d)?
A. a*(c + d)+ b(c + d)
B. a*(c + d)* + b(c + d)*
C. a*(c + d)+ b*(c + d)
D. (a + b)*c +(a + b)*d
Answer: D

A certain pipelined RISC machine has 8 general-purpose registers R0, R1, . . . , R7 and supports the following operations.
ADD Rs1, Rs2, Rd Add Rs1 to Rs2 and put the sum in Rd
MUL Rs1, Rs2, Rd Multiply Rs1 by Rs2 and put the product in Rd
An operation normally takes one cycle; however, an operation takes two cycles if it produces a result required by the immediately following operation in an operation sequence. Consider the expression AB + ABC + BC, where variables A, B, C are located in registers R0, R1, R2. If the contents of these three registers must not be modified, what is the minimum number of clock cycles required for an operation sequence that computes the value of AB + ABC + BC?
A. 5
B. 6
C. 7
D. 8
Answer: B

The Singleton design pattern is used to guarantee that only a single instance of a class may be instantiated. Which of the following is (are) true of this design pattern?
I. The Singleton class has a static factory method to provide its instance.
II. The Singleton class can be a subclass of another class.
III. The Singleton class has a private constructor.
A. I only
B. II only
C. III only
D. I, II, and III
Answer: D

A compiler generates code for the following assignment statement.
G := (A + B) * C - (D + E) * F
The target machine has a single accumulator and a single-address instruction set consisting of instructions load, store, add, subtract, and multiply. For the arithmetic operations, the left operand is taken from the accumulator and the result appears in the accumulator. The smallest possible number of instructions in the resulting code is
A. 5
B. 6
C. 7
D. 9
Answer: D

Consider a computer design in which multiple processors, each with a private cache memory, share global memory using a single bus. This bus is the critical system resource. Each processor can execute one instruction every 500 nanoseconds as long as memory references are satisfied by its local cache. When a cache miss occurs, the processor is delayed for an additional 2,000 nanoseconds. During half of this additional delay, the bus is dedicated to serving the cache miss. During the other half, the processor cannot continue, but the bus is free to service requests from other processors. On average, each instruction requires 2 memory references. On average, cache misses occur on 1 percent of references. What proportion of the capacity of the bus would a single processor consume, ignoring delays due to competition from other processors?
A. 1/50
B. 1/27
C. 1/25
D. 2/27
Answer: B

Let A be a finite set with m elements, and let B be a finite set with n elements. The number of distinct functions mapping A into B is
A. n^m
B. n!/(n - m)!
C. n!
D. n!/(m!(n - m)!)
Answer:
```
Answer: **A**

### 48. [high_school_physics]

```
The following are multiple choice questions (with answers) about high school physics.

Which of the following conditions will ensure that angular momentum is conserved? I. Conservation of linear momentum II. Zero net external force III. Zero net external torque
A. I and II only
B. I and III only
C. II and III only
D. III only
Answer: D

A pipe full of air is closed at one end. A standing wave is produced in the pipe, causing the pipe to sound a note. Which of the following is a correct statement about the wave’s properties at the closed end of the pipe?
A. The pressure is at a node, but the particle displacement is at an antinode.
B. The pressure is at an antinode, but the particle displacement is at a node.
C. The pressure and the particle displacement are both at nodes.
D. The pressure and the particle displacement are both at antinodes.
Answer: B

A photocell of work function ϕ = 2eV is connected to a resistor in series. Light of frequency f = 1 × 10^15 Hz hits a metal plate of the photocell. If the power of the light is P = 100 W, what is the current through the resistor?
A. 2:00 AM
B. 6:00 AM
C. 12:00 AM
D. 24 A
Answer: D

A microwave oven is connected to an outlet, 120 V, and draws a current of 2 amps. At what rate is energy being used by the microwave oven?
A. 10 W
B. 30 W
C. 60 W
D. 240 W
Answer: D

A point charge, Q = +1 mC, is fixed at the origin. How much work is required to move a charge, Q = +8 µC, from the point (0, 4 meters) to the point (3 meters, 0)?
A. 3.5 J
B. 6.0 J
C. 22.5 J
D. 40 J
Answer: B

A disk of radius 0.1 m initially at rest undergoes an angular acceleration of 2.0 rad/s^2. If the disk only rotates, find the total distance traveled by a point on the rim of the disk in 4.0 s.
A. 0.4 m
B. 0.8 m
C. 1.2 m
D. 1.6 m
Answer:
```
Answer: **D**

### 49. [professional_psychology]

```
The following are multiple choice questions (with answers) about professional psychology.

One of your therapy clients asks your advice about a good weight- reduction program. You have investigated the programs in the community and are enrolled in the one you consider the best. This program offers a $50 bonus to its patrons for each new person they bring into the program. Under these circumstances, your most appropriate response would be to
A. tell your client the pros and cons of each program you know about except for the one in which you are enrolled
B. recommend to your client the program in which you are enrolled and explain the $50 bonus you will receive
C. recommend to your client the program in which you are enrolled and offer to have the $50 bonus credited to your client's account in the program
D. tell your client the pros and cons of each program you know about, but do not claim the $50 bonus if your client enrolls in your program
Answer: D

There are three ways to measure the Central Tendency: the Mean, the Median and the Mode. From your knowledge about them, what is the mode?
A. less sensitive to extreme scores than the mean
B. more useful for skewed distributions
C. sensitive to extreme values and highly skewed distributions
D. the most frequently occurring number
Answer: D

In terms of Hofstede’s (1980) five cultural dimensions, the United States scores at the top of the scale on:
A. individualism.
B. individualism and power distance.
C. power distance and masculinity.
D. uncertainty avoidance.
Answer: A

Carl Jung believed that a client's transference:
A. is a fantasy that distracts the client from reality.
B. represents “mixed feelings” toward the therapist. 
C. "is a form of ""acting out."""
D. reflects the client’s personal and collective unconscious.
Answer: D

In the construction of a multiple regression equation for purposes of prediction, the optimal combination of measures is one in which the predictors
A. are uncorrelated with each other but are moderately correlated with the criterion
B. have low correlations with each other and low correlations with the criterion
C. are highly intercorrelated with each other and moderately correlated with the criterion
D. have low correlations with the criterion bur are moderately correlated with each other
Answer: A

Identify the only construct that is not pertinent to developmental models on intelligence:
A. Investment theory
B. The positive manifold
C. G theory
D. Primary mental ability theory
Answer:
```
Answer: **A**

### 50. [high_school_government_and_politics]

```
The following are multiple choice questions (with answers) about high school government and politics.

Uncertainty over the limits to presidential power is caused primarily by the fact that
A. the constitutional definition of those powers is broad and unspecific
B. most people agree that the Constitution places too many limits on presidential power
C. the Supreme Court consistently refuses to rule on cases concerning presidential powers
D. constitutional amendments have greatly increased presidential powers
Answer: A

The term "budget deficit" refers to the
A. annual increase in federal spending on the military
B. amount of interest on the national debt
C. difference between the initial budget proposals made by the president and Congress
D. amount the government spends in excess of its revenues
Answer: D

Which of the following cases established the precedent that a defendant must be informed of the right to remain silent, the right to a lawyer, and protection from self-incrimination?
A. Weeks v. United States
B. Betts v. Brady
C. Mapp v. Ohio
D. Miranda v. Arizona
Answer: D

Which of the following statements about cabinet departments is FALSE?
A. They are established by the legislative branch.
B. Their members often don't have much influence over presidential decisions.
C. They cannot all be run by leaders who belong to the same political party the president does.
D. Not every federal agency is a cabinet department.
Answer: C

Which of the following best states an argument made by James Madison in The Federalist number 10?
A. Honest politicians can prevent factions from developing.
B. Factions are more likely to occur in large republics than in small ones.
C. The negative effects of factionalism can be reduced by a republican government.
D. Free elections are the people's best defense against factionalism.
Answer: C

Which of the following statements does NOT accurately describe voting behavior in the United States?
A. Registered voters between the ages of 35 and 45 are more likely to vote than are those under the age of 21.
B. A registered voter who has attained his or her General Educational Development (GED) is less likely to vote than a high school dropout.
C. Registered voters are more likely to vote in general elections than they are in primary elections.
D. More women than men have voted in every presidential election since 1980.
Answer:
```
Answer: **B**

### 51. [moral_scenarios]

```
The following are multiple choice questions (with answers) about moral scenarios.

For which of these two scenarios does the main character (who uses I/me/my) do something clearly morally wrong, according to ordinary moral standards in the US as of 2020? Scenario 1 | I really loved my newborn baby boy. Scenario 2 | The new girl is embarrassed to live in a trailer park, so I invite the class to her house.
A. Wrong, Wrong
B. Wrong, Not wrong
C. Not wrong, Wrong
D. Not wrong, Not wrong
Answer: C

For which of these two scenarios does the main character (who uses I/me/my) do something clearly morally wrong, according to ordinary moral standards in the US as of 2020? Scenario 1 | I asked for a cigarette from the guy outside the store. Scenario 2 | When the security guard left his post, I pocketed a rare coin.
A. Wrong, Wrong
B. Wrong, Not wrong
C. Not wrong, Wrong
D. Not wrong, Not wrong
Answer: C

For which of these two scenarios does the main character (who uses I/me/my) do something clearly morally wrong, according to ordinary moral standards in the US as of 2020? Scenario 1 | I bought an old pick-up truck for work. Scenario 2 | I looked up at the gorgeous waterfall and threw a rock at it.
A. Wrong, Wrong
B. Wrong, Not wrong
C. Not wrong, Wrong
D. Not wrong, Not wrong
Answer: D

For which of these two scenarios does the main character (who uses I/me/my) do something clearly morally wrong, according to ordinary moral standards in the US as of 2020? Scenario 1 | During the house's demolition, I pushed him into a wall. Scenario 2 | When I choose a babysitter for my twin nieces, I don't care about her racial background.
A. Wrong, Wrong
B. Wrong, Not wrong
C. Not wrong, Wrong
D. Not wrong, Not wrong
Answer: B

For which of these two scenarios does the main character (who uses I/me/my) do something clearly morally wrong, according to ordinary moral standards in the US as of 2020? Scenario 1 | I took his wallet after he harassed me at the bar. Scenario 2 | I made love to my newborn baby boy.
A. Wrong, Wrong
B. Wrong, Not wrong
C. Not wrong, Wrong
D. Not wrong, Not wrong
Answer: A

For which of these two scenarios does the main character (who uses I/me/my) do something clearly morally wrong, according to ordinary moral standards in the US as of 2020? Scenario 1 | When I checked out from the hotel I turned in the dirty towels. Scenario 2 | I always kept my whiskey bottles out of reach of my 3-year-old son.
A. Wrong, Wrong
B. Wrong, Not wrong
C. Not wrong, Wrong
D. Not wrong, Not wrong
Answer:
```
Answer: **D**

### 52. [professional_law]

```
The following are multiple choice questions (with answers) about professional law.

A state legislature has recently enacted a statute making it a misdemeanor to curse or revile or use obscene or opprobrious language toward or in reference to a police officer perfonning his duties. A student at a state university organized a demonstration on campus to protest the war. The rally was attended by a group of 50 students who shouted anti-war messages at cars passing by. To show his contempt for the United States, the student sewed the American flag to the rear of his jeans. When a police officer saw the flag sown on the student's jeans, he approached and told him to remove the flag or he would be placed under arrest. The student became angered and shouted at the police officer, "Listen, you bastard, I'll wear this rag anywhere I please. " The student was subsequently placed under arrest and charged with violating the state statute. The student subsequently brings suit in state court challenging the constitutionality of the statute. The strongest constitutional argument for the student is that
A. the statute is void for vagueness under the Fourteenth Amendment's due process clause.
B. the statute is invalid because it violates the petitioner's freedom of speech under the First Amendment.
C. the statute is an abridgment of freedom of speech under the First Amendment because less restrictive means are available for achieving the same purpose.
D. the statute is overbroad and consequently invalid under the First and FourteenthAmendments.
Answer: D

A state has recently enacted a statute prohibiting the disposal of any nuclear wastes within the state. This law does not contravene or conflict with any federal statutes. A man operates a company in the state that is engaged in the disposal of nuclear wastes. Subsequent to the passage of the state statute, the man, not yet aware of the new law, entered into contracts with many out-of-state firms to dispose of their nuclear wastes in the state. On account of this new law, however, the man will be unable to perform these contracts. Assume that the man has standing to challenge this state law. Which of the following presents his strongest constitutional grounds to challenge the state law prohibiting the disposal of nuclear wastes within the state?
A. The commerce clause.
B. The equal protection clause of the Fourteenth Amendment.
C. The privileges and immunities clause of Article IV, Section 2. 
D. The contract clause.
Answer: A

Judge took judicial notice of some facts at the beginning of the trial. Which of the following is not an appropriate kind of fact for judicial notice?
A. Indisputable facts.
B. Facts that have been asserted by individual political organizations.
C. Facts recognized to be true by common knowledge.
D. Facts capable of scientific verification.
Answer: B

On October 1, 1980, a developer, owner of several hundred acres in a rural county, drafted a general development plan for the area. The duly recorded plan imposed elaborate limitations and restrictions upon the land in the plan, which was to be developed as a residential district. The restrictions were to extend to all persons acquiring any of the lots and to their heirs, assigns, and lessees. It was further provided that all subsequent owners would be charged with due notice of the restrictions. Among those restrictions in the general plan were the following:(22) A franchise right is created in a strip of land 10 feet in width along the rear of each lot for the use of public utility companies with right of ingress and egress. (23) No house or structure of any kind shall be built on the aforementioned strip of land running through the said blocks. In 2000, a retiree purchased one of the lots, built a house, and erected a fence in the rear of his property within the restricted area. In 2004, a teacher purchased a lot adjacent to the retiree's property and built a new house. Two years later, a librarian purchased the lot that adjoined the teacher's property. The three deeds to those properties each contained references to the deed book where the general plan was recorded. In 2008, the librarian began the construction of a seven-foot post-and-rail fence along the line dividing his lot with the teacher's, and along the center of the area subject to the franchise right. Although the teacher objected to its construction, the fence was completed. If the teacher seeks a mandatory injunction to compel removal of the librarian's fence, the court will most likely
A. grant relief, because the fence was in violation of the easement restriction. 
B. grant relief, because the encroachment of the fence violated the restriction in the original plan. 
C. deny relief, because the teacher failed to enforce the restriction against the retiree. 
D. deny relief, because the fence would not be construed as "a structure" within the terms of the restriction. 
Answer: B

A son owed a creditor $5,000. The son's father contacted the creditor and told him that he wanted to pay the son's debt. The father signed a document that stated the father would pay the son's debt at a rate of $500 a month for 10 months. The creditor made no written or oral commitment to forbear to sue the son to collect the $5,000 debt, and the father made no oral or written request for any such forbearance. For the next five months, the father made and the creditor accepted the $500 monthly payments as agreed. During that period, the creditor, in fact, did forbear to take any legal action against the son. However, the father then informed the creditor that he would make no further payments on the debt. Which of the following is the most persuasive argument that the father is liable to the creditor under the terms of their agreement?
A. The father's promise and the creditor's reliance thereon, if proved, gave rise to a valid claim by the creditor against the father based on the doctrine of promissory estoppel. 
B. Because it was foreseeable that the father's promise would induce the creditor to forbear taking any action against the son, such forbearance was, as a matter of law, a bargained-for consideration for the father's promise. 
C. The father's five payments to the creditor totaling $2,500 manifested a serious intent on the father's part to be contractually bound, and such manifestation is generally recognized as an effective substitute for consideration. 
D. By assuming the antecedent debt obligation that the son owed to the creditor, the father became a surety whose promise to the creditor was enforceable, since it was in writing and supported by adequate consideration. 
Answer: A

A state imposes a graduated income tax upon net income calculated under federal law. Section 22 of the city's reform act of 2010, which is to become effective on January 1, 2010, provides: "Any parent or guardian financially responsible for the education of his ward may claim a direct tax credit against his income tax liability equal to the amount of tuition of a child or children of high school age who does not attend a public high school. " Other provisions define "tuition" very broadly but limit the credit to tuition paid to schools meeting the educational requirements as determined by the state. On December 1, 2009, the parents of two students who attend a public high school in the state sue for a declaratory judgment and injunction in federal court, claiming that Section 22 violates the establishment clause of the First Amendment. The federal court will most likely
A. dismiss the action, because the suit involves a political question.
B. dismiss the action, because the plaintiffs lack standing as taxpayers.
C. dismiss the action, because the issues are not ripe.
D. dismiss the action, because the issues being litigated are moot.
Answer:
```
Answer: **C**

### 53. [high_school_government_and_politics]

```
The following are multiple choice questions (with answers) about high school government and politics.

Uncertainty over the limits to presidential power is caused primarily by the fact that
A. the constitutional definition of those powers is broad and unspecific
B. most people agree that the Constitution places too many limits on presidential power
C. the Supreme Court consistently refuses to rule on cases concerning presidential powers
D. constitutional amendments have greatly increased presidential powers
Answer: A

The term "budget deficit" refers to the
A. annual increase in federal spending on the military
B. amount of interest on the national debt
C. difference between the initial budget proposals made by the president and Congress
D. amount the government spends in excess of its revenues
Answer: D

Which of the following cases established the precedent that a defendant must be informed of the right to remain silent, the right to a lawyer, and protection from self-incrimination?
A. Weeks v. United States
B. Betts v. Brady
C. Mapp v. Ohio
D. Miranda v. Arizona
Answer: D

Which of the following statements about cabinet departments is FALSE?
A. They are established by the legislative branch.
B. Their members often don't have much influence over presidential decisions.
C. They cannot all be run by leaders who belong to the same political party the president does.
D. Not every federal agency is a cabinet department.
Answer: C

Which of the following best states an argument made by James Madison in The Federalist number 10?
A. Honest politicians can prevent factions from developing.
B. Factions are more likely to occur in large republics than in small ones.
C. The negative effects of factionalism can be reduced by a republican government.
D. Free elections are the people's best defense against factionalism.
Answer: C

A member of the president's cabinet is said to have "gone native" when that cabinet member
A. resigns to take a position as a consultant to lobbying groups
B. cedes control of his or her department to lifelong bureaucrats within the department
C. places his or her department's priorities above the president's
D. accepts bribes or expensive favors from businesses regulated by his or her department
Answer:
```
Answer: **C**

### 54. [professional_accounting]

```
The following are multiple choice questions (with answers) about professional accounting.

Box a nongovernmental not-for-profit organization had the following transactions during the year: Proceeds from sale of investments $80000 Purchase of property plant and equipment $10000 Proceeds from long-term debt $100000 Loss on sale of investment $5000 What amount should be reported as net cash provided by financing activities in Box's statement of cash flows?
A. $70,000
B. $75,000
C. $80,000
D. 100000
Answer: D

One hundred years ago, your great-great-grandmother invested $100 at 5% yearly interest. What is the investment worth today?
A. $13,000
B. $600
C. $15,000
D. $28,000
Answer: A

Krete is an unmarried taxpayer with income exclusively from wages. By December 31, year 1, Krete's employer has withheld $16,000 in federal income taxes and Krete has made no estimated tax payments. On April 15, year 2, Krete timely filed for an extension request to file her individual tax return, and paid $300 of additional taxes. Krete's year 1 tax liability was $16,500 when she timely filed her return on April 30, year 2, and paid the remaining tax liability balance. What amount would be subject to the penalty for underpayment of estimated taxes?
A. $0
B. $500
C. $1,650
D. $16,500
Answer: A

On January 1, year 1, Alpha Co. signed an annual maintenance agreement with a software provider for $15,000 and the maintenance period begins on March 1, year 2. Alpha also incurred $5,000 of costs on January 1, year 1, related to software modification requests that will increase the functionality of the software. Alpha depreciates and amortizes its computer and software assets over five years using the straight-line method. What amount is the total expense that Alpha should recognize related to the maintenance agreement and the software modifications for the year ended December 31, year 1?
A. $5,000
B. $13,500
C. $16,000
D. $20,000
Answer: B

An auditor traces the serial numbers on equipment to a nonissuer’s subledger. Which of the following management assertions is supported by this test?
A. Valuation and allocation
B. Completeness
C. Rights and obligations
D. Presentation and disclosure
Answer: B

Order the following (risk, return) pairs from least to most favourable, assuming the perspective of a rational and risk-averse investor: (2,2),(2,3) and (4,2). Risk is measured in standard deviations and return in percentage. HINT: Imagine a scatter diagram with standard deviation on the x-axis and return on the y-axis.
A. (4,2),(2,2),(2,3)
B. (2,2),(2,3),(4,2)
C. (2,2),(4,2),(2,3)
D. (2,3),(2,2),(4,2)
Answer:
```
Answer: **A**

### 55. [jurisprudence]

```
The following are multiple choice questions (with answers) about jurisprudence.

Which position does Rawls claim is the least likely to be adopted by the POP (people in the original position)?
A. The POP would choose equality above liberty.
B. The POP would opt for the 'maximin' strategy.
C. The POP would opt for the 'difference principle'.
D. The POP would reject the 'system of natural liberty.'
Answer: A

Functions of the law include all but which of the following?
A. maximizing individual freedom
B. providing a basis for compromise
C. keeping the peace
D. promoting the principles of the free enterprise system
Answer: D

Which word best summarizes Weber's explanation of the development of formally rational law?
A. Authority.
B. Charisma.
C. Co-operation.
D. Capitalism.
Answer: D

The ________ School of jurisprudence postulates that the law is based on what is "correct."
A. Natural Law
B. Analytical
C. Historical
D. Sociological
Answer: A

Iverson Jewelers wrote a letter to Miller, 'We have received an exceptionally fine self winding Rolox watch which we will sell to you at a very favorable price.'
A. The letter is an offer to sell
B. A valid offer cannot be made by letter.
C. The letter contains a valid offer which will terminate within a reasonable time.
D. The letter lacks one of the essential elements of an offer.
Answer: D

What is the main reason why many Critical Race Theorists are hostile to concepts such as 'justice', 'neutrality', and 'truth'?
A. Because of their vagueness.
B. Because they have failed to produce racial justice.
C. Because they are unrelated to contemporary conditions.
D. Because these ideas are alien to most people of colour.
Answer:
```
Answer: **B**

### 56. [international_law]

```
The following are multiple choice questions (with answers) about international law.

What types of force does Article 2(4) of the UN Charter prohibit?
A. Article 2(4) encompasses only armed force
B. Article 2(4) encompasses all types of force, including sanctions
C. Article 2(4) encompasses all interference in the domestic affairs of States
D. Article 2(4) encompasses force directed only against a State's territorial integrity
Answer: A

What is the judge ad hoc?
A. If a party to a contentious case before the ICJ does not have a national sitting as judge, it is entitled to nominate someone as a judge solely for that case, with the title of judge ad hoc
B. Judge ad hoc is the member of the bench of the ICJ with a casting vote
C. Judge ad hoc is a surrogate judge, in case a judge is disqualified or passes away
D. Judge ad hoc is the judge that each party will always nominate in every contentious case
Answer: A

Would a reservation to the definition of torture in the ICCPR be acceptable in contemporary practice?
A. This is an acceptable reservation if the reserving country's legislation employs a different definition
B. This is an unacceptable reservation because it contravenes the object and purpose of the ICCPR
C. This is an unacceptable reservation because the definition of torture in the ICCPR is consistent with customary international law
D. This is an acceptable reservation because under general international law States have the right to enter reservations to treaties
Answer: B

When 'consent' can serve as a circumstance precluding the wrongfulness of a State conduct?
A. Consent can serve as a circumstance precluding the wrongfulness whenever it is given
B. Consent can never serve as a circumstance precluding wrongfulness
C. Consent can serve as a circumstance precluding wrongfulness, provided the consent is valid and to the extent that the conduct remains within the limits of the consent given
D. Consent can always serve as a circumstance precluding wrongfulness, no matter which organ of the State gives it
Answer: C

How the consent to be bound of a State may be expressed?
A. The consent of a State to be bound is expressed only by ratification
B. The consent of a state to be bound by a treaty may be expressed by signature, ratification, acceptance, approval or accession
C. The consent of a State to be bound is expressed by signature
D. The consent of a State to be bound is expressed by whatever means they choose
Answer: B

What is personal (ratione personae) immunity?
A. Personal immunity is afforded to all physical persons
B. Personal immunity is that which is afforded in a personal capacity and hence does not cover conduct of the State as such
C. Personal immunity is afforded only to particular persons irrespective if their conduct was undertaken in a private or public capacity
D. Personal immunity is afforded to State officials for conduct undertaken in a public capacity
Answer:
```
Answer: **C**

### 57. [international_law]

```
The following are multiple choice questions (with answers) about international law.

What types of force does Article 2(4) of the UN Charter prohibit?
A. Article 2(4) encompasses only armed force
B. Article 2(4) encompasses all types of force, including sanctions
C. Article 2(4) encompasses all interference in the domestic affairs of States
D. Article 2(4) encompasses force directed only against a State's territorial integrity
Answer: A

What is the judge ad hoc?
A. If a party to a contentious case before the ICJ does not have a national sitting as judge, it is entitled to nominate someone as a judge solely for that case, with the title of judge ad hoc
B. Judge ad hoc is the member of the bench of the ICJ with a casting vote
C. Judge ad hoc is a surrogate judge, in case a judge is disqualified or passes away
D. Judge ad hoc is the judge that each party will always nominate in every contentious case
Answer: A

Would a reservation to the definition of torture in the ICCPR be acceptable in contemporary practice?
A. This is an acceptable reservation if the reserving country's legislation employs a different definition
B. This is an unacceptable reservation because it contravenes the object and purpose of the ICCPR
C. This is an unacceptable reservation because the definition of torture in the ICCPR is consistent with customary international law
D. This is an acceptable reservation because under general international law States have the right to enter reservations to treaties
Answer: B

When 'consent' can serve as a circumstance precluding the wrongfulness of a State conduct?
A. Consent can serve as a circumstance precluding the wrongfulness whenever it is given
B. Consent can never serve as a circumstance precluding wrongfulness
C. Consent can serve as a circumstance precluding wrongfulness, provided the consent is valid and to the extent that the conduct remains within the limits of the consent given
D. Consent can always serve as a circumstance precluding wrongfulness, no matter which organ of the State gives it
Answer: C

How the consent to be bound of a State may be expressed?
A. The consent of a State to be bound is expressed only by ratification
B. The consent of a state to be bound by a treaty may be expressed by signature, ratification, acceptance, approval or accession
C. The consent of a State to be bound is expressed by signature
D. The consent of a State to be bound is expressed by whatever means they choose
Answer: B

Who is an 'injured State' in the law of international responsibility?
A. A State is 'injured' in case that it has suffered a damage from the internationally wrongful conduct
B. A State is 'injured' in cases that there has been a violation of a peremptory norm of international law
C. A State is 'injured' should it acknowledge the existence of the internationally wrongful conduct
D. A State is 'injured' if the obligation breached was owed to it individually or if it was owed to a group of States, including that State, and it was specially affected
Answer:
```
Answer: **D**

### 58. [professional_law]

```
The following are multiple choice questions (with answers) about professional law.

A state legislature has recently enacted a statute making it a misdemeanor to curse or revile or use obscene or opprobrious language toward or in reference to a police officer perfonning his duties. A student at a state university organized a demonstration on campus to protest the war. The rally was attended by a group of 50 students who shouted anti-war messages at cars passing by. To show his contempt for the United States, the student sewed the American flag to the rear of his jeans. When a police officer saw the flag sown on the student's jeans, he approached and told him to remove the flag or he would be placed under arrest. The student became angered and shouted at the police officer, "Listen, you bastard, I'll wear this rag anywhere I please. " The student was subsequently placed under arrest and charged with violating the state statute. The student subsequently brings suit in state court challenging the constitutionality of the statute. The strongest constitutional argument for the student is that
A. the statute is void for vagueness under the Fourteenth Amendment's due process clause.
B. the statute is invalid because it violates the petitioner's freedom of speech under the First Amendment.
C. the statute is an abridgment of freedom of speech under the First Amendment because less restrictive means are available for achieving the same purpose.
D. the statute is overbroad and consequently invalid under the First and FourteenthAmendments.
Answer: D

A state has recently enacted a statute prohibiting the disposal of any nuclear wastes within the state. This law does not contravene or conflict with any federal statutes. A man operates a company in the state that is engaged in the disposal of nuclear wastes. Subsequent to the passage of the state statute, the man, not yet aware of the new law, entered into contracts with many out-of-state firms to dispose of their nuclear wastes in the state. On account of this new law, however, the man will be unable to perform these contracts. Assume that the man has standing to challenge this state law. Which of the following presents his strongest constitutional grounds to challenge the state law prohibiting the disposal of nuclear wastes within the state?
A. The commerce clause.
B. The equal protection clause of the Fourteenth Amendment.
C. The privileges and immunities clause of Article IV, Section 2. 
D. The contract clause.
Answer: A

Judge took judicial notice of some facts at the beginning of the trial. Which of the following is not an appropriate kind of fact for judicial notice?
A. Indisputable facts.
B. Facts that have been asserted by individual political organizations.
C. Facts recognized to be true by common knowledge.
D. Facts capable of scientific verification.
Answer: B

On October 1, 1980, a developer, owner of several hundred acres in a rural county, drafted a general development plan for the area. The duly recorded plan imposed elaborate limitations and restrictions upon the land in the plan, which was to be developed as a residential district. The restrictions were to extend to all persons acquiring any of the lots and to their heirs, assigns, and lessees. It was further provided that all subsequent owners would be charged with due notice of the restrictions. Among those restrictions in the general plan were the following:(22) A franchise right is created in a strip of land 10 feet in width along the rear of each lot for the use of public utility companies with right of ingress and egress. (23) No house or structure of any kind shall be built on the aforementioned strip of land running through the said blocks. In 2000, a retiree purchased one of the lots, built a house, and erected a fence in the rear of his property within the restricted area. In 2004, a teacher purchased a lot adjacent to the retiree's property and built a new house. Two years later, a librarian purchased the lot that adjoined the teacher's property. The three deeds to those properties each contained references to the deed book where the general plan was recorded. In 2008, the librarian began the construction of a seven-foot post-and-rail fence along the line dividing his lot with the teacher's, and along the center of the area subject to the franchise right. Although the teacher objected to its construction, the fence was completed. If the teacher seeks a mandatory injunction to compel removal of the librarian's fence, the court will most likely
A. grant relief, because the fence was in violation of the easement restriction. 
B. grant relief, because the encroachment of the fence violated the restriction in the original plan. 
C. deny relief, because the teacher failed to enforce the restriction against the retiree. 
D. deny relief, because the fence would not be construed as "a structure" within the terms of the restriction. 
Answer: B

A son owed a creditor $5,000. The son's father contacted the creditor and told him that he wanted to pay the son's debt. The father signed a document that stated the father would pay the son's debt at a rate of $500 a month for 10 months. The creditor made no written or oral commitment to forbear to sue the son to collect the $5,000 debt, and the father made no oral or written request for any such forbearance. For the next five months, the father made and the creditor accepted the $500 monthly payments as agreed. During that period, the creditor, in fact, did forbear to take any legal action against the son. However, the father then informed the creditor that he would make no further payments on the debt. Which of the following is the most persuasive argument that the father is liable to the creditor under the terms of their agreement?
A. The father's promise and the creditor's reliance thereon, if proved, gave rise to a valid claim by the creditor against the father based on the doctrine of promissory estoppel. 
B. Because it was foreseeable that the father's promise would induce the creditor to forbear taking any action against the son, such forbearance was, as a matter of law, a bargained-for consideration for the father's promise. 
C. The father's five payments to the creditor totaling $2,500 manifested a serious intent on the father's part to be contractually bound, and such manifestation is generally recognized as an effective substitute for consideration. 
D. By assuming the antecedent debt obligation that the son owed to the creditor, the father became a surety whose promise to the creditor was enforceable, since it was in writing and supported by adequate consideration. 
Answer: A

A national association brought suit in federal court against a city. The suit charged the city with violating several federal statutes. During the trial, the association's attorney called the city attorney as an adverse witness. The city attorney asserted the attorney-client privilege and objected that the matters communicated to him were intended to be confidential. The issues raised by the objections of the city attorney should be resolved under
A. federal privilege law.
B. the privilege law of the forum state.
C. either federal privilege law or the privilege law of the forum state as determined by the court.
D. either federal privilege law or the privilege law of the forum state, whichever the court determines is more likely to admit the evidence.
Answer:
```
Answer: **A**

### 59. [marketing]

```
The following are multiple choice questions (with answers) about marketing.

_____________ is a natural outcome when combining demographic and geographic variables.
A. Geodemographics
B. Product differentiation.
C. ANSOFF matrix.
D. Brand management.
Answer: A

In an organization, the group of people tasked with buying decisions is referred to as the _______________.
A. Outsourcing unit.
B. Procurement centre.
C. Chief executive unit.
D. Decision-making unit.
Answer: D

Which of the following is an assumption in Maslow's hierarchy of needs?
A. Needs are dependent on culture and also on social class.
B. Lower-level needs must be at least partially satisfied before higher needs can affect behaviour.
C. Needs are not prioritized or arranged in any particular order.
D. Satisfied needs are motivators, and new needs emerge when current needs remain unmet.
Answer: B

The single group within society that is most vulnerable to reference group influence is:
A. The older consumer who feels somewhat left out of things.
B. The married women, many of whom feel a need for stability in their lives.
C. New immigrants who really want to assimilate into their new culture.
D. Children, who base most of their buying decisions on outside influences.
Answer: D

Although the content and quality can be as controlled as direct mail, response rates of this medium are lower because of the lack of a personal address mechanism. This media format is known as:
A. Care lines.
B. Direct mail.
C. Inserts.
D. Door to door.
Answer: D

There are three main types of buying situations in an organization, referred to by Robinson, Faris, and Wind (1967) as _____________.
A. Repeat purchases.
B. Buyphases.
C. Buyclasses.
D. Tenders.
Answer:
```
Answer: **C**

### 60. [moral_scenarios]

```
The following are multiple choice questions (with answers) about moral scenarios.

For which of these two scenarios does the main character (who uses I/me/my) do something clearly morally wrong, according to ordinary moral standards in the US as of 2020? Scenario 1 | I really loved my newborn baby boy. Scenario 2 | The new girl is embarrassed to live in a trailer park, so I invite the class to her house.
A. Wrong, Wrong
B. Wrong, Not wrong
C. Not wrong, Wrong
D. Not wrong, Not wrong
Answer: C

For which of these two scenarios does the main character (who uses I/me/my) do something clearly morally wrong, according to ordinary moral standards in the US as of 2020? Scenario 1 | I asked for a cigarette from the guy outside the store. Scenario 2 | When the security guard left his post, I pocketed a rare coin.
A. Wrong, Wrong
B. Wrong, Not wrong
C. Not wrong, Wrong
D. Not wrong, Not wrong
Answer: C

For which of these two scenarios does the main character (who uses I/me/my) do something clearly morally wrong, according to ordinary moral standards in the US as of 2020? Scenario 1 | I bought an old pick-up truck for work. Scenario 2 | I looked up at the gorgeous waterfall and threw a rock at it.
A. Wrong, Wrong
B. Wrong, Not wrong
C. Not wrong, Wrong
D. Not wrong, Not wrong
Answer: D

For which of these two scenarios does the main character (who uses I/me/my) do something clearly morally wrong, according to ordinary moral standards in the US as of 2020? Scenario 1 | During the house's demolition, I pushed him into a wall. Scenario 2 | When I choose a babysitter for my twin nieces, I don't care about her racial background.
A. Wrong, Wrong
B. Wrong, Not wrong
C. Not wrong, Wrong
D. Not wrong, Not wrong
Answer: B

For which of these two scenarios does the main character (who uses I/me/my) do something clearly morally wrong, according to ordinary moral standards in the US as of 2020? Scenario 1 | I took his wallet after he harassed me at the bar. Scenario 2 | I made love to my newborn baby boy.
A. Wrong, Wrong
B. Wrong, Not wrong
C. Not wrong, Wrong
D. Not wrong, Not wrong
Answer: A

For which of these two scenarios does the main character (who uses I/me/my) do something clearly morally wrong, according to ordinary moral standards in the US as of 2020? Scenario 1 | My mom asked me to deep clean the bath so I covered it in bleach. Scenario 2 | For the funeral, I wore my clown costume to honor the dead.
A. Wrong, Wrong
B. Wrong, Not wrong
C. Not wrong, Wrong
D. Not wrong, Not wrong
Answer:
```
Answer: **C**

### 61. [public_relations]

```
The following are multiple choice questions (with answers) about public relations.

What should a public relations media practitioner do if she does not know the answer to a reporter's question?
A. Give the reporter other information she is certain is correct.
B. Say that the information is 'off the record' and will be disseminated later.
C. Say 'I don't know' and promise to provide the information later.
D. Say 'no comment,' rather than appear uninformed.
Answer: C

In issues management, what is the most proactive approach to addressing negative or misleading information posted online about your organization?
A. Buy domain names that could be used by opposition groups.
B. Post anonymous comments on blogs to combat this information.
C. Prepare a news release that discredits the inaccurate information.
D. Make policy changes to address complaints highlighted on these sites.
Answer: D

Which of these statements is true of the Vatican in 2010 at the time of the accusations of child abuse cover-ups?
A. There was a coordinated media response.
B. Consistent messages were communicated.
C. Criticisms were taken as attacks on the Catholic Church.
D. The credibility of the Vatican was upheld.
Answer: C

At which stage in the planning process would a situation analysis be carried out?
A. Defining the program
B. Planning the program
C. Taking action and implementing ideas
D. Evaluation of the program
Answer: A

Earth Hour was a campaign launched by which organization?
A. Greenpeace
B. The UN
C. Oxfam
D. World Wildlife Fund
Answer: D

A ________ campaign occurs when people from two or more opposing sides of an argument have emotional convictions about a decision that has the power to impact their lives.
A. public relations
B. public issues
C. crisis management
D. consumer relations
Answer:
```
Answer: **B**

### 62. [professional_law]

```
The following are multiple choice questions (with answers) about professional law.

A state legislature has recently enacted a statute making it a misdemeanor to curse or revile or use obscene or opprobrious language toward or in reference to a police officer perfonning his duties. A student at a state university organized a demonstration on campus to protest the war. The rally was attended by a group of 50 students who shouted anti-war messages at cars passing by. To show his contempt for the United States, the student sewed the American flag to the rear of his jeans. When a police officer saw the flag sown on the student's jeans, he approached and told him to remove the flag or he would be placed under arrest. The student became angered and shouted at the police officer, "Listen, you bastard, I'll wear this rag anywhere I please. " The student was subsequently placed under arrest and charged with violating the state statute. The student subsequently brings suit in state court challenging the constitutionality of the statute. The strongest constitutional argument for the student is that
A. the statute is void for vagueness under the Fourteenth Amendment's due process clause.
B. the statute is invalid because it violates the petitioner's freedom of speech under the First Amendment.
C. the statute is an abridgment of freedom of speech under the First Amendment because less restrictive means are available for achieving the same purpose.
D. the statute is overbroad and consequently invalid under the First and FourteenthAmendments.
Answer: D

A state has recently enacted a statute prohibiting the disposal of any nuclear wastes within the state. This law does not contravene or conflict with any federal statutes. A man operates a company in the state that is engaged in the disposal of nuclear wastes. Subsequent to the passage of the state statute, the man, not yet aware of the new law, entered into contracts with many out-of-state firms to dispose of their nuclear wastes in the state. On account of this new law, however, the man will be unable to perform these contracts. Assume that the man has standing to challenge this state law. Which of the following presents his strongest constitutional grounds to challenge the state law prohibiting the disposal of nuclear wastes within the state?
A. The commerce clause.
B. The equal protection clause of the Fourteenth Amendment.
C. The privileges and immunities clause of Article IV, Section 2. 
D. The contract clause.
Answer: A

Judge took judicial notice of some facts at the beginning of the trial. Which of the following is not an appropriate kind of fact for judicial notice?
A. Indisputable facts.
B. Facts that have been asserted by individual political organizations.
C. Facts recognized to be true by common knowledge.
D. Facts capable of scientific verification.
Answer: B

On October 1, 1980, a developer, owner of several hundred acres in a rural county, drafted a general development plan for the area. The duly recorded plan imposed elaborate limitations and restrictions upon the land in the plan, which was to be developed as a residential district. The restrictions were to extend to all persons acquiring any of the lots and to their heirs, assigns, and lessees. It was further provided that all subsequent owners would be charged with due notice of the restrictions. Among those restrictions in the general plan were the following:(22) A franchise right is created in a strip of land 10 feet in width along the rear of each lot for the use of public utility companies with right of ingress and egress. (23) No house or structure of any kind shall be built on the aforementioned strip of land running through the said blocks. In 2000, a retiree purchased one of the lots, built a house, and erected a fence in the rear of his property within the restricted area. In 2004, a teacher purchased a lot adjacent to the retiree's property and built a new house. Two years later, a librarian purchased the lot that adjoined the teacher's property. The three deeds to those properties each contained references to the deed book where the general plan was recorded. In 2008, the librarian began the construction of a seven-foot post-and-rail fence along the line dividing his lot with the teacher's, and along the center of the area subject to the franchise right. Although the teacher objected to its construction, the fence was completed. If the teacher seeks a mandatory injunction to compel removal of the librarian's fence, the court will most likely
A. grant relief, because the fence was in violation of the easement restriction. 
B. grant relief, because the encroachment of the fence violated the restriction in the original plan. 
C. deny relief, because the teacher failed to enforce the restriction against the retiree. 
D. deny relief, because the fence would not be construed as "a structure" within the terms of the restriction. 
Answer: B

A son owed a creditor $5,000. The son's father contacted the creditor and told him that he wanted to pay the son's debt. The father signed a document that stated the father would pay the son's debt at a rate of $500 a month for 10 months. The creditor made no written or oral commitment to forbear to sue the son to collect the $5,000 debt, and the father made no oral or written request for any such forbearance. For the next five months, the father made and the creditor accepted the $500 monthly payments as agreed. During that period, the creditor, in fact, did forbear to take any legal action against the son. However, the father then informed the creditor that he would make no further payments on the debt. Which of the following is the most persuasive argument that the father is liable to the creditor under the terms of their agreement?
A. The father's promise and the creditor's reliance thereon, if proved, gave rise to a valid claim by the creditor against the father based on the doctrine of promissory estoppel. 
B. Because it was foreseeable that the father's promise would induce the creditor to forbear taking any action against the son, such forbearance was, as a matter of law, a bargained-for consideration for the father's promise. 
C. The father's five payments to the creditor totaling $2,500 manifested a serious intent on the father's part to be contractually bound, and such manifestation is generally recognized as an effective substitute for consideration. 
D. By assuming the antecedent debt obligation that the son owed to the creditor, the father became a surety whose promise to the creditor was enforceable, since it was in writing and supported by adequate consideration. 
Answer: A

One afternoon a motorist was driving her car when she struck a pedestrian who entered the intersection. The pedestrian was seriously injured in the accident and sued the motorist to recover damages for personal injuries. There were no witnesses to the accident. The motorist claims that the traffic light was green at the time she entered the intersection. The pedestrian now has amnesia and cannot remember how the accident occurred. At trial, the pedestrian's attorney calls a witness to testify. The witness proposes to testify that she was in the emergency room at the hospital when the pedestrian was brought in following the accident. The witness further testifies that while the pedestrian was being treated for his injuries, the nurse asked him, "How did you get hurt?" The pedestrian replied, "The car ran the red light and struck me as I was walking through the intersection. "Upon objection, this evidence is
A. admissible as a statement made for purposes of medical treatment and diagnosis.
B. admissible as a present sense impression.
C. inadmissible, because the witness was eavesdropping on a confidential communication between the pedestrian and a hospital representative.
D. inadmissible, because it is hearsay not within any recognized exception.
Answer:
```
Answer: **D**

### 63. [high_school_psychology]

```
The following are multiple choice questions (with answers) about high school psychology.

Ani believes that her attitudes and behavior play a central role in what happens to her. Such a belief is likely to be associated with
A. a strong superego.
B. low self-esteem.
C. low self-efficacy.
D. an internal locus of control.
Answer: D

According to Caplan's model of consultee-centered case consultation, the consultant is primarily interested in
A. identifying the causes and solutions of the client's presenting problems
B. identifying and eliminating the causes of the consultee's difficulties in handling a problem
C. establishing a hierarchy of authority to enable effective decision making
D. presenting a single, well-defined and unambiguous course of action for the consultant to overcome skills deficits
Answer: B

While swimming in the ocean, Ivan is frightened by a dark shadow in the water even before he has the chance to identify what the shadow is. The synaptic connections taking place during this incident of fright are best described by which of the following?
A. Messages are sent from the thalamus directly to the amygdala.
B. Messages are sent from the thalamus to the "what" and "where" pathways.
C. Messages are sent from the parasympathetic nervous system to the cerebral cortex.
D. Messages are sent from the frontal lobes to the pituitary gland.
Answer: A

According to the Individuals with Disabilities Education Improvement Act, which of the following must an educational agency do before it changes the educational placement of a student with a disability?
A. Give the child a trial period in the new environment
B. Notify the parents in writing
C. Obtain school board approval
D. Obtain parental consent
Answer: B

Pascale is interested in the processing strategies children use to learn new information. Pascale would best be classified as what type of psychologist?
A. sociocultural
B. clinical
C. cognitive
D. behaviorist
Answer: C

Which of the following samples would be considered most representative of male college students?
A. A group of thirty fraternity brothers from Penn State
B. A random sample taken between classes in the business wing of various universities
C. Sixty male members of each class from Princeton, Yale, Harvard, Dartmouth, and Columbia
D. Twenty male members of each class from a cross-section of colleges and universities
Answer:
```
Answer: **D**

### 64. [jurisprudence]

```
The following are multiple choice questions (with answers) about jurisprudence.

Which position does Rawls claim is the least likely to be adopted by the POP (people in the original position)?
A. The POP would choose equality above liberty.
B. The POP would opt for the 'maximin' strategy.
C. The POP would opt for the 'difference principle'.
D. The POP would reject the 'system of natural liberty.'
Answer: A

Functions of the law include all but which of the following?
A. maximizing individual freedom
B. providing a basis for compromise
C. keeping the peace
D. promoting the principles of the free enterprise system
Answer: D

Which word best summarizes Weber's explanation of the development of formally rational law?
A. Authority.
B. Charisma.
C. Co-operation.
D. Capitalism.
Answer: D

The ________ School of jurisprudence postulates that the law is based on what is "correct."
A. Natural Law
B. Analytical
C. Historical
D. Sociological
Answer: A

Iverson Jewelers wrote a letter to Miller, 'We have received an exceptionally fine self winding Rolox watch which we will sell to you at a very favorable price.'
A. The letter is an offer to sell
B. A valid offer cannot be made by letter.
C. The letter contains a valid offer which will terminate within a reasonable time.
D. The letter lacks one of the essential elements of an offer.
Answer: D

Which statement below is the least likely to follow logically from Dworkin's notion of law as integrity?
A. It is likely to generate more individual rights and greater liberty.
B. It renders a community more genuine.
C. It opens the door to authoritarianism.
D. It improves the moral justification for the exercise of political power.
Answer:
```
Answer: **C**

### 65. [medical_genetics]

```
The following are multiple choice questions (with answers) about medical genetics.

Large triplet repeat expansions can be detected by:
A. polymerase chain reaction.
B. single strand conformational polymorphism analysis.
C. Southern blotting.
D. Western blotting.
Answer: C

DNA ligase is
A. an enzyme that joins fragments in normal DNA replication
B. an enzyme of bacterial origin which cuts DNA at defined base sequences
C. an enzyme that facilitates transcription of specific genes
D. an enzyme which limits the level to which a particular nutrient reaches
Answer: A

A gene showing codominance
A. has both alleles independently expressed in the heterozygote
B. has one allele dominant to the other
C. has alleles tightly linked on the same chromosome
D. has alleles expressed at the same time in development
Answer: A

Which of the following conditions does not show multifactorial inheritance?
A. Pyloric stenosis
B. Schizophrenia
C. Spina bifida (neural tube defects)
D. Marfan syndrome
Answer: D

The stage of meiosis in which chromosomes pair and cross over is:
A. prophase I
B. metaphase I
C. prophase II
D. metaphase II
Answer: A

X-chromosome inactivation
A. results in genetically turning off one of the two X chromosomes in female mammals
B. takes place in humans so that the same X chromosome is inactive in all of the cells of a female
C. is the cause of the Y chromosome being genetically inactive
D. occurs in fruit flies but not in mammals
Answer:
```
Answer: **A**

### 66. [anatomy]

```
The following are multiple choice questions (with answers) about anatomy.

What is the embryological origin of the hyoid bone?
A. The first pharyngeal arch
B. The first and second pharyngeal arches
C. The second pharyngeal arch
D. The second and third pharyngeal arches
Answer: D

Which of these branches of the trigeminal nerve contain somatic motor processes?
A. The supraorbital nerve
B. The infraorbital nerve
C. The mental nerve
D. None of the above
Answer: D

The pleura
A. have no sensory innervation.
B. are separated by a 2 mm space.
C. extend into the neck.
D. are composed of respiratory epithelium.
Answer: C

In Angle's Class II Div 2 occlusion there is
A. excess overbite of the upper lateral incisors.
B. negative overjet of the upper central incisors.
C. excess overjet of the upper lateral incisors.
D. excess overjet of the upper central incisors.
Answer: C

Which of the following is the body cavity that contains the pituitary gland?
A. Abdominal
B. Cranial
C. Pleural
D. Spinal
Answer: B

In the fetus, the ductus arteriosus passes blood from the
A. pulmonary vein to the aorta.
B. aorta to pulmonary vein.
C. pulmonary artery to the aorta.
D. aorta to the pulmonary artery.
Answer:
```
Answer: **C**

### 67. [human_sexuality]

```
The following are multiple choice questions (with answers) about human sexuality.

Morning sickness is typically a problem:
A. during the first trimester
B. during the second trimester
C. during the third trimester
D. all through the pregnancy
Answer: A

A woman who knows she has active herpes and untreated syphilis but continues to have sex without informing her partners of her condition has, in psychoanalytic terms:
A. a strong ego
B. a weak superego
C. a weak id
D. a strong superego
Answer: B

Women's ability to have multiple orgasms is primarily due to:
A. the fact that they do not have a refractory period.
B. the response of the inner layers of the vagina.
C. having alternating orgasms in different locations.
D. the G-Spot.
Answer: A

The nature of homosexual activities that occur during preadolescence include all but which of the following?
A. sexual intercourse
B. circle jerks
C. exhibitionism
D. touching each other's genitals
Answer: A

The most common disorder among men who seek sexual therapy is:
A. premature ejaculation
B. inhibited ejaculation
C. erectile disorder
D. ejaculatory disorder
Answer: C

The follicular phase is to the __________ as the luteal phase is to the secretory phase.
A. postovulatory
B. menstrual
C. proliferative
D. myometrial
Answer:
```
Answer: **C**

### 68. [computer_security]

```
The following are multiple choice questions (with answers) about computer security.

SHA-1 has a message digest of
A. 160 bits
B. 512 bits
C. 628 bits
D. 820 bits
Answer: A

_____________ can modify data on your system – so that your system doesn’t run correctly or you can no longer access specific data, or it may even ask for ransom in order to give your access.
A. IM – Trojans
B. Backdoor Trojans
C. Trojan-Downloader
D. Ransom Trojan
Answer: D

What is ethical hacking?
A. "Hacking" ethics so they justify unintended selfish behavior
B. Hacking systems (e.g., during penetration testing) to expose vulnerabilities so they can be fixed, rather than exploited
C. Hacking into systems run by those whose ethics you disagree with
D. A slang term for rapid software development, e.g., as part of hackathons
Answer: B

Exploitation of the Heartbleed bug permits
A. overwriting cryptographic keys in memory
B. a kind of code injection
C. a read outside bounds of a buffer
D. a format string attack
Answer: C

The ____________ is anything which your search engine cannot search.
A. Haunted web
B. World Wide Web
C. Surface web
D. Deep Web
Answer: D

Buffer-overflow may remain as a bug in apps if __________ are not done fully.
A. boundary hacks
B. memory checks
C. boundary checks
D. buffer checks
Answer:
```
Answer: **C**

### 69. [college_computer_science]

```
The following are multiple choice questions (with answers) about college computer science.

Which of the following regular expressions is equivalent to (describes the same set of strings as) (a* + b)*(c + d)?
A. a*(c + d)+ b(c + d)
B. a*(c + d)* + b(c + d)*
C. a*(c + d)+ b*(c + d)
D. (a + b)*c +(a + b)*d
Answer: D

A certain pipelined RISC machine has 8 general-purpose registers R0, R1, . . . , R7 and supports the following operations.
ADD Rs1, Rs2, Rd Add Rs1 to Rs2 and put the sum in Rd
MUL Rs1, Rs2, Rd Multiply Rs1 by Rs2 and put the product in Rd
An operation normally takes one cycle; however, an operation takes two cycles if it produces a result required by the immediately following operation in an operation sequence. Consider the expression AB + ABC + BC, where variables A, B, C are located in registers R0, R1, R2. If the contents of these three registers must not be modified, what is the minimum number of clock cycles required for an operation sequence that computes the value of AB + ABC + BC?
A. 5
B. 6
C. 7
D. 8
Answer: B

The Singleton design pattern is used to guarantee that only a single instance of a class may be instantiated. Which of the following is (are) true of this design pattern?
I. The Singleton class has a static factory method to provide its instance.
II. The Singleton class can be a subclass of another class.
III. The Singleton class has a private constructor.
A. I only
B. II only
C. III only
D. I, II, and III
Answer: D

A compiler generates code for the following assignment statement.
G := (A + B) * C - (D + E) * F
The target machine has a single accumulator and a single-address instruction set consisting of instructions load, store, add, subtract, and multiply. For the arithmetic operations, the left operand is taken from the accumulator and the result appears in the accumulator. The smallest possible number of instructions in the resulting code is
A. 5
B. 6
C. 7
D. 9
Answer: D

Consider a computer design in which multiple processors, each with a private cache memory, share global memory using a single bus. This bus is the critical system resource. Each processor can execute one instruction every 500 nanoseconds as long as memory references are satisfied by its local cache. When a cache miss occurs, the processor is delayed for an additional 2,000 nanoseconds. During half of this additional delay, the bus is dedicated to serving the cache miss. During the other half, the processor cannot continue, but the bus is free to service requests from other processors. On average, each instruction requires 2 memory references. On average, cache misses occur on 1 percent of references. What proportion of the capacity of the bus would a single processor consume, ignoring delays due to competition from other processors?
A. 1/50
B. 1/27
C. 1/25
D. 2/27
Answer: B

Let T be a depth-first search tree of a connected undirected graph G. For each vertex v of T, let pre(v) be the number of nodes visited up to and including v during a preorder traversal of T, and post(v) be the number of nodes visited up to and including v during a postorder traversal of T. The lowest common ancestor of vertices u and v in T is a vertex w of T such that w is an ancestor of both u and v, and no child of w is an ancestor of both u and v. Let (u, v) be an edge in G that is not in T, such that pre(u) < pre(v). Which of the following statements about u and v must be true?
I. post(u) < post(v)
II. u is an ancestor of v in T.
III. If w is the lowest common ancestor of u and v in T, then w = u.
A. I only
B. II only
C. III only
D. II and III
Answer:
```
Answer: **D**

### 70. [high_school_world_history]

```
The following are multiple choice questions (with answers) about high school world history.

This question refers to the following information.
He contains all works and desires and all perfumes and all tastes. He enfolds the whole universe and in silence is loving to all. This is the Spirit that is in my heart, this is Brahman. To him I shall come when I go beyond this life, and to him will come he who has faith and doubts not.
—The Upanishads, India, c. 1000 BCE
To which religion does the speaker most likely belong?
A. Hinduism
B. Buddhism
C. Shintoism
D. Zoroastrianism
Answer: A

This question refers to the following information.
"The struggle against neo-colonialism is not aimed at excluding the capital of the developed world from operating in less developed countries. It is aimed at preventing the financial power of the developed countries being used in such a way as to impoverish the less developed.
Non-alignment, as practiced by Ghana and many other countries, is based on co-operation with all States whether they be capitalist, socialist or have a mixed economy. Such a policy, therefore, involves foreign investment from capitalist countries, but it must be invested in accordance with a national plan drawn up by the government of the non-aligned State with its own interests in mind. The issue is not what return the foreign investor receives on his investments…The question is one of power. A State in the grip of neo-colonialism is not master of its own destiny."
Kwame Nkrumah, Neo-Colonialism, 1965
Which of the following provides the best context for Nkrumah's writings?
A. The Industrial Revolution
B. Decolonization
C. Regional Free Trade Associations
D. Autarky
Answer: B

This question refers to the following information.
"The real grievance of the worker is the insecurity of his existence; he is not sure that he will always have work, he is not sure that he will always be healthy, and he foresees that he will one day be old and unfit to work. If he falls into poverty, even if only through a prolonged illness, he is then completely helpless, exam_ins to his own devices, and society does not currently recognize any real obligation towards him beyond the usual help for the poor, even if he has been working all the time ever so faithfully and diligently. The usual help for the poor, however, leaves a lot to be desired, especially in large cities, where it is very much worse than in the country."
Otto von Bismarck, 1884
Otto von Bismarck likely made this speech in reaction to which of the following issues?
A. Social acceptance of child labor
B. Declining life expectancy in Germany
C. Criticisms of German trade tariffs
D. Negative effects attributed to industrial capitalism
Answer: D

This question refers to the following information.
"Indeed, as both the fatwas of distinguished [scholars] who base their opinion on reason and tradition alike and the consensus of the Sunni community agree that the ancient obligation of extirpation, extermination, and expulsion of evil innovation must be the aim of our exalted aspiration, for "Religious zeal is a victory for the Faith of God the Beneficent"; then, in accordance with the words of the Prophet (Peace upon him!) "Whosoever introduces evil innovation into our order must be expelled" and "Whosoever does aught against our order must be expelled," action has become necessary and exigent…"
Letter from Ottoman Sultan Selim I to Safavid Shah Ismail I, 1514
The letter from Selim I is most clearly an example of which of the following?
A. The maintenance of military supremacy at all costs
B. Expanding tensions between religious sects
C. Factors that brought about the collapse of the Ottoman Empire
D. Peacemaking efforts among the Islamic empires
Answer: B

This question refers to the following information.
"At least one of the [world's] societies would have to somehow enormously increase its productivity [in order to achieve global hegemony]. That quantum jump would have to be made before the various scientific, technological, agricultural, and industrial revolutions on which our post-quantum-leap world rests. It could only be accomplished by exploiting the ecosystems, mineral resources, and human assets of whole continents outside the lands of the society making the jump. Western Europe did just that by means of its brutality and guns and, more important, by geographical and ecological luck."
Copyright © 2015 Cambridge University Press.
Alfred Crosby, historian, Ecological Imperialism, 2004
The "quantum jump" mentioned in the passage most directly contributed to which of the following developments in the period 1450–1750 C.E.?
A. A breakdown in trade routes through the collapse of the established state structure
B. An increase in the population of the world through more plentiful supplies of food
C. The spread of Chinese and Indian belief systems across the world
D. An increase in social unrest
Answer: B

This question refers to the following information.
Every two months His Majesty sends from Lima 60,000 pesos to pay for the mita of the Indians. Up on the Huanacavelica range there are 3,000 or 4,000 Indians working in the mercury mine, with picks and hammers, breaking up the ore. And when they have filled up their little sacks, the poor fellows, loaded down, climb up those ladders and rigging, so distressing that a man can hardly get up them. That is the way they work in this mine, with many lights and the loud noise of the pounding and great confusion. Nor is that the greatest evil; that is due to thievish and undisciplined superintendents. According to His Majesty's warrant, the mine owners at Potosí have a right to the mita of 13,300 Indians. These mita Indians earn each day 4 reals. Besides these there are others not under obligation, who hire themselves out voluntarily: these each get from 12 to 16 reals, and some up to 24, according to how well they wield their picks or their reputation for knowing how to get the ore out.
Antonio Vasquez de Espinosa, report on mining in Huanacavelica and Potosí, 1620s
The third principal reason the local Yakut and Tungus natives are ruined is that from the time they first came under Russian control, they have been forced to pay yasak tribute. Some have paid in sables, others in red foxes, still others in cash. At first there were plenty of furbearing animals there, but now there are no sables and not many foxes in those lands, from the shores of the Arctic Ocean all the way south to the great Lena River. Moreover, almost half the natives cannot hunt because they no longer have horses, many of which have been pawned to the yasak collectors.
Heinrich von Füch, "On the Treatment of Natives in Northeast Siberia," 1744
Both passages are best seen in the context of which of the following broad developments?
A. Emerging systems of coerced labor
B. The expansion of chattel slavery
C. Increased reliance on prison-camp workforces
D. Colonization of overseas territories by joint-stock companies
Answer:
```
Answer: **A**

### 71. [elementary_mathematics]

```
The following are multiple choice questions (with answers) about elementary mathematics.

The population of the city where Michelle was born is 145,826. What is the value of the 5 in the number 145,826?
A. 5 thousands
B. 5 hundreds
C. 5 tens
D. 5 ones
Answer: A

Olivia used the rule "Add 11" to create the number pattern shown below. 10, 21, 32, 43, 54 Which statement about the number pattern is true?
A. The 10th number in the pattern will be an even number.
B. The number pattern will never have two even numbers next to each other.
C. The next two numbers in the pattern will be an even number then an odd number.
D. If the number pattern started with an odd number then the pattern would have only odd numbers in it.
Answer: B

A total of 30 players will play basketball at a park. There will be exactly 5 players on each team. Which statement correctly explains how to find the number of teams needed?
A. Add 5 to 30 to find 35 teams.
B. Divide 30 by 5 to find 6 teams.
C. Multiply 30 and 5 to find 150 teams.
D. Subtract 5 from 30 to find 25 teams.
Answer: B

A store sells 107 different colors of paint. They have 25 cans of each color in storage. The number of cans of paint the store has in storage can be found using the expression below. 107 × 25. How many cans of paint does the store have in storage?
A. 749
B. 2,675
C. 2,945
D. 4,250
Answer: B

Which expression is equivalent to 5 x 9?
A. (5 x 4) x (6 x 5)
B. (5 x 5) + (5 x 4)
C. (5 x 5) + (5 x 9)
D. (5 x 9) x (6 x 9)
Answer: B

A soccer team has $90.00 to buy soccer balls. If one soccer ball costs $15.60, what is the greatest number of soccer balls the team can buy?
A. 4
B. 5
C. 6
D. 7
Answer:
```
Answer: **B**

### 72. [human_aging]

```
The following are multiple choice questions (with answers) about human aging.

Which of the following persons is more likely to remain at home alone, as of 2019?
A. An Asian man or woman
B. A Hispanic man
C. An African American woman
D. A white man or woman
Answer: C

The finding that adults tend to remember events from their adolescence better than from other periods in their lives is referred to as the
A. Adolescence advantage
B. Reminiscence bump
C. Memorial memorial
D. Quadratic retrieval spike
Answer: B

When older adults move to a new state after retirement, which of the following is the more likely destination?
A. Texas
B. California
C. Hawaii
D. Vermont
Answer: A

Which element in tobacco smoke is responsible for cancers?
A. Nicotine
B. Tar
C. Carbon monoxide
D. Smoke particles
Answer: B

All other things being equal, which of the following persons is more likely to show osteoporosis?
A. An older Hispanic American woman
B. An older African American woman
C. An older Asian American woman
D. An older Native American woman
Answer: C

Current estimates of how many people exercise on a regular basis suggest that _____ do
A. Less than 50%
B. More than 75%
C. Less than 25%
D. More than 50%
Answer:
```
Answer: **A**

### 73. [high_school_physics]

```
The following are multiple choice questions (with answers) about high school physics.

Which of the following conditions will ensure that angular momentum is conserved? I. Conservation of linear momentum II. Zero net external force III. Zero net external torque
A. I and II only
B. I and III only
C. II and III only
D. III only
Answer: D

A pipe full of air is closed at one end. A standing wave is produced in the pipe, causing the pipe to sound a note. Which of the following is a correct statement about the wave’s properties at the closed end of the pipe?
A. The pressure is at a node, but the particle displacement is at an antinode.
B. The pressure is at an antinode, but the particle displacement is at a node.
C. The pressure and the particle displacement are both at nodes.
D. The pressure and the particle displacement are both at antinodes.
Answer: B

A photocell of work function ϕ = 2eV is connected to a resistor in series. Light of frequency f = 1 × 10^15 Hz hits a metal plate of the photocell. If the power of the light is P = 100 W, what is the current through the resistor?
A. 2:00 AM
B. 6:00 AM
C. 12:00 AM
D. 24 A
Answer: D

A microwave oven is connected to an outlet, 120 V, and draws a current of 2 amps. At what rate is energy being used by the microwave oven?
A. 10 W
B. 30 W
C. 60 W
D. 240 W
Answer: D

A point charge, Q = +1 mC, is fixed at the origin. How much work is required to move a charge, Q = +8 µC, from the point (0, 4 meters) to the point (3 meters, 0)?
A. 3.5 J
B. 6.0 J
C. 22.5 J
D. 40 J
Answer: B

An object carries a charge of –1 C. How many excess electrons does it contain?
A. 6.25 × 10^18
B. 8.00 × 10^18
C. 1.60 × 10^19
D. 3.20 × 10^19
Answer:
```
Answer: **A**

### 74. [philosophy]

```
The following are multiple choice questions (with answers) about philosophy.

Psychological egoism is:
A. an ethical theory about how we ought to behave.
B. a generalization concerning the way people tend to behave.
C. a claim about human nature and the ways people are capable of behaving.
D. none of the above.
Answer: C

According to Moore’s “ideal utilitarianism,” the right action is the one that brings about the greatest amount of:
A. pleasure.
B. happiness.
C. good.
D. virtue.
Answer: C

According to d'Holbach, people always act according to _____.
A. free choices
B. dictates of the soul
C. necessary natural laws
D. undetermined will
Answer: C

Before Tolstoy's Christian conversion, what was his perspective on the meaning of life?
A. optimist
B. satisfied
C. nominally religious
D. pessimist
Answer: D

The study of reality in the broadest sense, an inquiry into the elemental nature of the universe and the things in it, is known as _____.
A. metaphysics
B. epistemology
C. quantum physics
D. axiology
Answer: A

According to Socrates, it is important that we discover what makes a particular action (e.g., a merciful or just act) the kind of action that it is, because without such knowledge:
A. no one in society will ever do any action that really is merciful or just, only those actions that they think are merciful or just.
B. the primary purpose of human existence--which is to think and to know--is replaced by a focus on morality (acting and doing).
C. we can refer only to how people characterize actions without knowing why such actions should be characterized that way.
D. there would be no way to distinguish one kind of action (e.g., a merciful action) from another kind of action (e.g., a just action).
Answer:
```
Answer: **C**

### 75. [professional_law]

```
The following are multiple choice questions (with answers) about professional law.

A state legislature has recently enacted a statute making it a misdemeanor to curse or revile or use obscene or opprobrious language toward or in reference to a police officer perfonning his duties. A student at a state university organized a demonstration on campus to protest the war. The rally was attended by a group of 50 students who shouted anti-war messages at cars passing by. To show his contempt for the United States, the student sewed the American flag to the rear of his jeans. When a police officer saw the flag sown on the student's jeans, he approached and told him to remove the flag or he would be placed under arrest. The student became angered and shouted at the police officer, "Listen, you bastard, I'll wear this rag anywhere I please. " The student was subsequently placed under arrest and charged with violating the state statute. The student subsequently brings suit in state court challenging the constitutionality of the statute. The strongest constitutional argument for the student is that
A. the statute is void for vagueness under the Fourteenth Amendment's due process clause.
B. the statute is invalid because it violates the petitioner's freedom of speech under the First Amendment.
C. the statute is an abridgment of freedom of speech under the First Amendment because less restrictive means are available for achieving the same purpose.
D. the statute is overbroad and consequently invalid under the First and FourteenthAmendments.
Answer: D

A state has recently enacted a statute prohibiting the disposal of any nuclear wastes within the state. This law does not contravene or conflict with any federal statutes. A man operates a company in the state that is engaged in the disposal of nuclear wastes. Subsequent to the passage of the state statute, the man, not yet aware of the new law, entered into contracts with many out-of-state firms to dispose of their nuclear wastes in the state. On account of this new law, however, the man will be unable to perform these contracts. Assume that the man has standing to challenge this state law. Which of the following presents his strongest constitutional grounds to challenge the state law prohibiting the disposal of nuclear wastes within the state?
A. The commerce clause.
B. The equal protection clause of the Fourteenth Amendment.
C. The privileges and immunities clause of Article IV, Section 2. 
D. The contract clause.
Answer: A

Judge took judicial notice of some facts at the beginning of the trial. Which of the following is not an appropriate kind of fact for judicial notice?
A. Indisputable facts.
B. Facts that have been asserted by individual political organizations.
C. Facts recognized to be true by common knowledge.
D. Facts capable of scientific verification.
Answer: B

On October 1, 1980, a developer, owner of several hundred acres in a rural county, drafted a general development plan for the area. The duly recorded plan imposed elaborate limitations and restrictions upon the land in the plan, which was to be developed as a residential district. The restrictions were to extend to all persons acquiring any of the lots and to their heirs, assigns, and lessees. It was further provided that all subsequent owners would be charged with due notice of the restrictions. Among those restrictions in the general plan were the following:(22) A franchise right is created in a strip of land 10 feet in width along the rear of each lot for the use of public utility companies with right of ingress and egress. (23) No house or structure of any kind shall be built on the aforementioned strip of land running through the said blocks. In 2000, a retiree purchased one of the lots, built a house, and erected a fence in the rear of his property within the restricted area. In 2004, a teacher purchased a lot adjacent to the retiree's property and built a new house. Two years later, a librarian purchased the lot that adjoined the teacher's property. The three deeds to those properties each contained references to the deed book where the general plan was recorded. In 2008, the librarian began the construction of a seven-foot post-and-rail fence along the line dividing his lot with the teacher's, and along the center of the area subject to the franchise right. Although the teacher objected to its construction, the fence was completed. If the teacher seeks a mandatory injunction to compel removal of the librarian's fence, the court will most likely
A. grant relief, because the fence was in violation of the easement restriction. 
B. grant relief, because the encroachment of the fence violated the restriction in the original plan. 
C. deny relief, because the teacher failed to enforce the restriction against the retiree. 
D. deny relief, because the fence would not be construed as "a structure" within the terms of the restriction. 
Answer: B

A son owed a creditor $5,000. The son's father contacted the creditor and told him that he wanted to pay the son's debt. The father signed a document that stated the father would pay the son's debt at a rate of $500 a month for 10 months. The creditor made no written or oral commitment to forbear to sue the son to collect the $5,000 debt, and the father made no oral or written request for any such forbearance. For the next five months, the father made and the creditor accepted the $500 monthly payments as agreed. During that period, the creditor, in fact, did forbear to take any legal action against the son. However, the father then informed the creditor that he would make no further payments on the debt. Which of the following is the most persuasive argument that the father is liable to the creditor under the terms of their agreement?
A. The father's promise and the creditor's reliance thereon, if proved, gave rise to a valid claim by the creditor against the father based on the doctrine of promissory estoppel. 
B. Because it was foreseeable that the father's promise would induce the creditor to forbear taking any action against the son, such forbearance was, as a matter of law, a bargained-for consideration for the father's promise. 
C. The father's five payments to the creditor totaling $2,500 manifested a serious intent on the father's part to be contractually bound, and such manifestation is generally recognized as an effective substitute for consideration. 
D. By assuming the antecedent debt obligation that the son owed to the creditor, the father became a surety whose promise to the creditor was enforceable, since it was in writing and supported by adequate consideration. 
Answer: A

A woman leased a condo from the owner for a period of one year. After six months, the owner gave the woman a written option to purchase the condo for $100,000 before the expiration of the lease. With the owner's approval, the woman spent $10,000 to have the kitchen remodeled to her personal specifications. One month before the end of the lease, the owner notified the woman that he was revoking the option to purchase the condo. One week later, the woman delivered a written notice of acceptance of the option, but the owner refused to sell. If the woman initiates suit for breach of contract, which of the following is her strongest argument that an enforceable contract was formed between her and the owner?
A. Because the woman had until the expiration of the lease to accept the offer, the owner's revocation would be ineffective.
B. Because the owner was a merchant, the written offer was irrevocable for a period not exceeding three months.
C. Because the owner's offer invited a return promise as acceptance, the woman manifested her intent to accept by remodeling the kitchen.
D. After the woman paid to have the kitchen remodeled, an option contract resulted, because the owner knew the woman was relying on the offer to her detriment.
Answer:
```
Answer: **D**

### 76. [high_school_macroeconomics]

```
The following are multiple choice questions (with answers) about high school macroeconomics.

Which of the following is not included in the U.S. GDP?
A. The U.S. military opens a new base in a foreign country with 1000 U.S. personnel.
B. Japanese consumers buy thousands of CDs produced in the United States.
C. An American pop singer performs a sold-out concert in Paris.
D. A French theatrical production tours dozens of American cities.
Answer: C

The short-run Phillips curve indicates a
A. direct relation between unemployment and inflation
B. direct relation between price and quantity demanded
C. inverse relation between price and quantity demanded
D. inverse relation between unemployment and inflation
Answer: D

A federal deficit occurs when
A. exports exceed imports.
B. imports exceed exports.
C. federal tax collections exceed spending.
D. federal spending exceeds federal tax revenues.
Answer: D

Holding all else equal which of the following monetary policies would be used to boost U.S. exports?
A. Increasing the discount rate
B. Increasing the reserve ratio
C. Buying government securities
D. Lowering tariffs
Answer: C

Which of the following policies best describes supply-side fiscal policy?
A. An increase in the money supply
B. Increased government spending
C. Lower taxes on research and development of new technology
D. Higher taxes on household income
Answer: C

What does the presence of discouraged workers do to the measurement of the unemployment rate?
A. Discouraged workers are counted as "out of the labor force" thus the unemployment rate is understated making the economy look stronger than it is.
B. Discouraged workers are counted as "out of the labor force" thus the unemployment rate is understated making the economy look weaker than it is.
C. Discouraged workers are not surveyed so there is no impact on the unemployment rate.
D. Discouraged workers are counted as "unemployed" thus the unemployment rate is understated making the economy look stronger than it is.
Answer:
```
Answer: **A**

### 77. [miscellaneous]

```
The following are multiple choice questions (with answers) about miscellaneous.

How many axles does a standard automobile have?
A. one
B. two
C. four
D. eight
Answer: B

What place is named in the title of the 1979 live album by rock legends Cheap Trick?
A. Budapest
B. Budokan
C. Bhutan
D. Britain
Answer: B

Who is the shortest man to ever win an NBA slam dunk competition?
A. Anthony 'Spud' Webb
B. Michael 'Air' Jordan
C. Tyrone 'Muggsy' Bogues
D. Julius 'Dr J' Erving
Answer: A

What is produced during photosynthesis?
A. hydrogen
B. nylon
C. oxygen
D. light
Answer: C

Which of these songs was a Top 10 hit for the rock band The Police?
A. 'Radio Ga-Ga'
B. 'Ob-la-di Ob-la-da'
C. 'De Do Do Do De Da Da Da'
D. 'In-a-Gadda-Da-Vida'
Answer: C

In the film 'The Talented Mr Ripley' who plays Mr Ripley?
A. Jude Law
B. Matt Damon
C. Dustin Hoffman
D. Ben Affleck
Answer:
```
Answer: **B**

### 78. [anatomy]

```
The following are multiple choice questions (with answers) about anatomy.

What is the embryological origin of the hyoid bone?
A. The first pharyngeal arch
B. The first and second pharyngeal arches
C. The second pharyngeal arch
D. The second and third pharyngeal arches
Answer: D

Which of these branches of the trigeminal nerve contain somatic motor processes?
A. The supraorbital nerve
B. The infraorbital nerve
C. The mental nerve
D. None of the above
Answer: D

The pleura
A. have no sensory innervation.
B. are separated by a 2 mm space.
C. extend into the neck.
D. are composed of respiratory epithelium.
Answer: C

In Angle's Class II Div 2 occlusion there is
A. excess overbite of the upper lateral incisors.
B. negative overjet of the upper central incisors.
C. excess overjet of the upper lateral incisors.
D. excess overjet of the upper central incisors.
Answer: C

Which of the following is the body cavity that contains the pituitary gland?
A. Abdominal
B. Cranial
C. Pleural
D. Spinal
Answer: B

Which of the following allows gas exchange in the lungs?
A. Alveoli
B. Bronchi
C. Bronchioles
D. Capillaries
Answer:
```
Answer: **A**

### 79. [philosophy]

```
The following are multiple choice questions (with answers) about philosophy.

Psychological egoism is:
A. an ethical theory about how we ought to behave.
B. a generalization concerning the way people tend to behave.
C. a claim about human nature and the ways people are capable of behaving.
D. none of the above.
Answer: C

According to Moore’s “ideal utilitarianism,” the right action is the one that brings about the greatest amount of:
A. pleasure.
B. happiness.
C. good.
D. virtue.
Answer: C

According to d'Holbach, people always act according to _____.
A. free choices
B. dictates of the soul
C. necessary natural laws
D. undetermined will
Answer: C

Before Tolstoy's Christian conversion, what was his perspective on the meaning of life?
A. optimist
B. satisfied
C. nominally religious
D. pessimist
Answer: D

The study of reality in the broadest sense, an inquiry into the elemental nature of the universe and the things in it, is known as _____.
A. metaphysics
B. epistemology
C. quantum physics
D. axiology
Answer: A

Anscombe criticizes as absurd Kant’s idea of:
A. the thing in itself.
B. the categorical imperative.
C. the phenomenal self.
D. legislating for oneself.
Answer:
```
Answer: **D**

### 80. [formal_logic]

```
The following are multiple choice questions (with answers) about formal logic.

Select the best translation into predicate logic: No people drive on Mars.
A. ~Pd
B. (∀x)(Px ∨ ~Dx)
C. (∀x)(Px ⊃ ~Dx)
D. ~Dp
Answer: C

Select the best translation into predicate logic.George borrows Hector's lawnmower. (g: George; h: Hector; l: Hector's lawnmower; Bxyx: x borrows y from z)
A. Blgh
B. Bhlg
C. Bglh
D. Bghl
Answer: C

Select the best English interpretation of the given arguments in predicate logic.
Dm
(∀x)(Wx ⊃ ~Dx)
(∀x)Wx ∨ Ag	/ (∃x)Ax
A. Marina is a dancer. Some weaklings are not dancers. Either everything is a weakling or Georgia plays volleyball. So something plays volleyball.
B. Marina is a dancer. No weakling is a dancer. Everything is either a weakling or plays volleyball. So something plays volleyball.
C. Marina is a dancer. Some weaklings are not dancers. Everything is either a weakling or plays volleyball. So something plays volleyball.
D. Marina is a dancer. No weakling is a dancer. Either everything is a weakling or Georgia plays volleyball. So something plays volleyball.
Answer: D

Construct a complete truth table for the following pairs of propositions. Then, using the truth tables, determine whether the statements are logically equivalent or contradictory. If neither, determine whether they are consistent or inconsistent. Justify your answers.
E ⊃ (F · E) and ~E · F
A. Logically equivalent
B. Contradictory
C. Neither logically equivalent nor contradictory, but consistent
D. Inconsistent
Answer: C

Which of the given formulas of PL is the best symbolization of the following sentence?
Turtles live long lives and are happy creatures, unless they are injured.
A. (L • H) ≡ I
B. (L • H) ∨ I
C. L • (H ∨ I)
D. L • (H ⊃ R)
Answer: B

Select the best translation into predicate logic. Some firefighters rescue victims from danger. (Dx: x is danger; Fx: x is a firefighter; Vx: x is a victim; Rxyz: x rescues y from z)
A. (∃x){Fx • (∀y)[Dy ⊃ (∃z)(Vz • Rxyz)]}
B. (∃x){Fx • (∃y)[Vy • (∃z)(Dz • Rxyz)]}
C. (∃x){Fx • (∃y)(∃z)[(Vy • Dz) • Rxzy]}
D. (∀x){Fx ⊃ (∃y)[Vy . (∃z)(Dz • Rxyz)]}
Answer:
```
Answer: **B**

### 81. [professional_medicine]

```
The following are multiple choice questions (with answers) about professional medicine.

A 42-year-old man comes to the office for preoperative evaluation prior to undergoing adrenalectomy scheduled in 2 weeks. One month ago, he received care in the emergency department for pain over his right flank following a motor vehicle collision. At that time, blood pressure was 160/100 mm Hg and CT scan of the abdomen showed an incidental 10-cm left adrenal mass. Results of laboratory studies, including complete blood count, serum electrolyte concentrations, and liver function tests, were within the reference ranges. The patient otherwise had been healthy and had never been told that he had elevated blood pressure. He takes no medications. A follow-up visit in the office 2 weeks ago disclosed elevated urinary normetanephrine and metanephrine and plasma aldosterone concentrations. The patient was referred to a surgeon, who recommended the adrenalectomy. Today, vital signs are temperature 36.6°C (97.9°F), pulse 100/min, respirations 14/min, and blood pressure 170/95 mm Hg. Physical examination discloses no significant findings. Initial preoperative preparation should include treatment with which of the following?
A. Labetalol
B. A loading dose of potassium chloride
C. Nifedipine
D. Phenoxybenzamine
Answer: D

A 36-year-old male presents to the office with a 3-week history of low back pain. He denies any recent trauma but says that he climbs in and out of his truck numerous times a day for his job. Examination of the patient in the prone position reveals a deep sacral sulcus on the left, a posterior inferior lateral angle on the right, and a lumbosacral junction that springs freely on compression. The most likely diagnosis is
A. left-on-left sacral torsion
B. left-on-right sacral torsion
C. right unilateral sacral flexion
D. right-on-right sacral torsion
Answer: D

A previously healthy 32-year-old woman comes to the physician 8 months after her husband was killed in a car crash. Since that time, she has had a decreased appetite and difficulty falling asleep. She states that she is often sad and cries frequently. She has been rechecking the door lock five times before leaving her house and has to count exactly five pieces of toilet paper before she uses it. She says that she has always been a perfectionist but these urges and rituals are new. Pharmacotherapy should be targeted to which of the following neurotransmitters?
A. Dopamine
B. Glutamate
C. Norepinephrine
D. Serotonin
Answer: D

A 44-year-old man comes to the office because of a 3-day history of sore throat, nonproductive cough, runny nose, and frontal headache. He says the headache is worse in the morning and ibuprofen does provide some relief. He has not had shortness of breath. Medical history is unremarkable. He takes no medications other than the ibuprofen for pain. Vital signs are temperature 37.4°C (99.4°F), pulse 88/min, respirations 18/min, and blood pressure 120/84 mm Hg. Examination of the nares shows erythematous mucous membranes. Examination of the throat shows erythema and follicular lymphoid hyperplasia on the posterior oropharynx. There is no palpable cervical adenopathy. Lungs are clear to auscultation. Which of the following is the most likely cause of this patient's symptoms?
A. Allergic rhinitis
B. Epstein-Barr virus
C. Mycoplasma pneumoniae
D. Rhinovirus
Answer: D

A 22-year-old male marathon runner presents to the office with the complaint of right-sided rib pain when he runs long distances. Physical examination reveals normal heart and lung findings and an exhalation dysfunction at ribs 4-5 on the right. Which of the following muscles or muscle groups will be most useful in correcting this dysfunction utilizing a direct method?
A. anterior scalene
B. latissimus dorsi
C. pectoralis minor
D. quadratus lumborum
Answer: C

During a clinical study examining the effects of exercise, men between the ages of 20 and 30 years are evaluated during a 15- minute session on a treadmill. The average pulse for the last 2 minutes of the session is 175/min. During the last minute of exercise, various measurements are taken. Compared with the measurement before the session, which of the following is most likely to be decreased?
A. Pulse pressure
B. Stroke volume
C. Systolic blood pressure
D. Total peripheral resistance
Answer:
```
Answer: **D**

### 82. [prehistory]

```
The following are multiple choice questions (with answers) about prehistory.

The great Mayan king Pacal built temples in the city of Palenque in order to:
A. satisfy the powerful Mayan astronomer priests.
B. display his generosity to the common people, since they were allowed to live in the temples.
C. frighten away enemies, in particular the Spaniards.
D. legitimize his kingship, since his father was not royal.
Answer: D

According to Timothy Pauketat, the evidence for social stratification and political power at Cahokia suggests:
A. a center of Mississippian civilization with conditions similar to the rise of early states.
B. the limitations of authority in a Native American society of egalitarian foragers.
C. a simple chiefdom or perhaps a complex chiefdom had evolved by A.D. 1500.
D. a center of Mississippian civilization with conditions similar to societies on the Northwest Coast of North America.
Answer: A

Researchers now believe that the decline of the Maya was caused chiefly by:
A. a cataclysm of some kind, such as an earthquake, volcano, or tsunami.
B. ecological degradation resulting from slash-and-burn farming techniques.
C. endless wars between neighboring Mayan city-states.
D. practices of interbreeding that led to a steep rise in congenital disorders.
Answer: B

Recent research on hominid species dating from the Middle Pliocene indicates there was (as of 2020):
A. a great amount of species diversity, or a single species that exhibited a lot of diversity.
B. very little species diversity during this period and very few hominids.
C. decreased species diversity due to a prolonged ice age followed by a severe drought.
D. decreased species diversity but increased numbers of hammerstones and flakes, indicating stone tool manufacture.
Answer: A

What is the approximate mean cranial capacity of Homo erectus?
A. under 650 cc
B. about 800 cc
C. just under 1000 cc
D. 1200 cc
Answer: C

Which of the following was described as “the slow agency of existing causes”?
A. natural selection
B. catastrophism
C. uniformitarianism
D. unilineal evolution
Answer:
```
Answer: **C**

### 83. [elementary_mathematics]

```
The following are multiple choice questions (with answers) about elementary mathematics.

The population of the city where Michelle was born is 145,826. What is the value of the 5 in the number 145,826?
A. 5 thousands
B. 5 hundreds
C. 5 tens
D. 5 ones
Answer: A

Olivia used the rule "Add 11" to create the number pattern shown below. 10, 21, 32, 43, 54 Which statement about the number pattern is true?
A. The 10th number in the pattern will be an even number.
B. The number pattern will never have two even numbers next to each other.
C. The next two numbers in the pattern will be an even number then an odd number.
D. If the number pattern started with an odd number then the pattern would have only odd numbers in it.
Answer: B

A total of 30 players will play basketball at a park. There will be exactly 5 players on each team. Which statement correctly explains how to find the number of teams needed?
A. Add 5 to 30 to find 35 teams.
B. Divide 30 by 5 to find 6 teams.
C. Multiply 30 and 5 to find 150 teams.
D. Subtract 5 from 30 to find 25 teams.
Answer: B

A store sells 107 different colors of paint. They have 25 cans of each color in storage. The number of cans of paint the store has in storage can be found using the expression below. 107 × 25. How many cans of paint does the store have in storage?
A. 749
B. 2,675
C. 2,945
D. 4,250
Answer: B

Which expression is equivalent to 5 x 9?
A. (5 x 4) x (6 x 5)
B. (5 x 5) + (5 x 4)
C. (5 x 5) + (5 x 9)
D. (5 x 9) x (6 x 9)
Answer: B

A car gets 24 miles per gallon of gasoline (mi/gal). How many gallons of gasoline would the car need to travel 144 miles?
A. 6.5 gallons
B. 6 gallons
C. 5.5 gallons
D. 5 gallons
Answer:
```
Answer: **B**

### 84. [high_school_psychology]

```
The following are multiple choice questions (with answers) about high school psychology.

Ani believes that her attitudes and behavior play a central role in what happens to her. Such a belief is likely to be associated with
A. a strong superego.
B. low self-esteem.
C. low self-efficacy.
D. an internal locus of control.
Answer: D

According to Caplan's model of consultee-centered case consultation, the consultant is primarily interested in
A. identifying the causes and solutions of the client's presenting problems
B. identifying and eliminating the causes of the consultee's difficulties in handling a problem
C. establishing a hierarchy of authority to enable effective decision making
D. presenting a single, well-defined and unambiguous course of action for the consultant to overcome skills deficits
Answer: B

While swimming in the ocean, Ivan is frightened by a dark shadow in the water even before he has the chance to identify what the shadow is. The synaptic connections taking place during this incident of fright are best described by which of the following?
A. Messages are sent from the thalamus directly to the amygdala.
B. Messages are sent from the thalamus to the "what" and "where" pathways.
C. Messages are sent from the parasympathetic nervous system to the cerebral cortex.
D. Messages are sent from the frontal lobes to the pituitary gland.
Answer: A

According to the Individuals with Disabilities Education Improvement Act, which of the following must an educational agency do before it changes the educational placement of a student with a disability?
A. Give the child a trial period in the new environment
B. Notify the parents in writing
C. Obtain school board approval
D. Obtain parental consent
Answer: B

Pascale is interested in the processing strategies children use to learn new information. Pascale would best be classified as what type of psychologist?
A. sociocultural
B. clinical
C. cognitive
D. behaviorist
Answer: C

A school psychologist is providing feedback to a student's parents regarding the student's performance on a measure of academic achievement. To explain the concept of grade equivalent, the school psychologist should explain that it is
A. the average score on that measure obtained by students in a given grade
B. the average score on that measure obtained by students at a given age
C. the grade in which a student should be placed in school
D. utilized to determine accountability among peers
Answer:
```
Answer: **A**

### 85. [us_foreign_policy]

```
The following are multiple choice questions (with answers) about us foreign policy.

How did the 2008 financial crisis affect America's international reputation?
A. It damaged support for the US model of political economy and capitalism
B. It created anger at the United States for exaggerating the crisis
C. It increased support for American global leadership under President Obama
D. It reduced global use of the US dollar
Answer: A

How did NSC-68 change U.S. strategy?
A. It globalized containment.
B. It militarized containment.
C. It called for the development of the hydrogen bomb.
D. All of the above
Answer: D

The realm of policy decisions concerned primarily with relations between the United States and the rest of the world is known as
A. terrorism policy.
B. economic policy.
C. foreign policy.
D. international policy.
Answer: C

How do Defensive Realism and Offensive Realism differ in their explanation of state behaviour?
A. Defensive realists place greater emphasis on the role of international institutions
B. Defensive realists place less emphasis on geographical factors
C. Offensive realists give more priority to the national interest than Defensive realists.
D. Defensive realists believe states are security maximizers, while Offensive realists believe states to be power maximizers
Answer: D

How did Donald Trump attack globalization in the 2016 campaign?
A. Globalization had made men like him too rich
B. Globalization only benefited certain American states, such as New York
C. Liberal elites had encouraged globalization, while 'ordinary Americans' lost jobs because of it
D. Globalization encouraged damaging trade wars
Answer: C

Global and regional international trade agreements work by using which of the following mechanisms?
A. Reciprocity across multiple issues
B. Reputational concerns of the actors
C. Side payments for adjusting to the organization (such as the Common Agricultural Policy in the EU)
D. ALL of the above
Answer:
```
Answer: **D**

### 86. [prehistory]

```
The following are multiple choice questions (with answers) about prehistory.

The great Mayan king Pacal built temples in the city of Palenque in order to:
A. satisfy the powerful Mayan astronomer priests.
B. display his generosity to the common people, since they were allowed to live in the temples.
C. frighten away enemies, in particular the Spaniards.
D. legitimize his kingship, since his father was not royal.
Answer: D

According to Timothy Pauketat, the evidence for social stratification and political power at Cahokia suggests:
A. a center of Mississippian civilization with conditions similar to the rise of early states.
B. the limitations of authority in a Native American society of egalitarian foragers.
C. a simple chiefdom or perhaps a complex chiefdom had evolved by A.D. 1500.
D. a center of Mississippian civilization with conditions similar to societies on the Northwest Coast of North America.
Answer: A

Researchers now believe that the decline of the Maya was caused chiefly by:
A. a cataclysm of some kind, such as an earthquake, volcano, or tsunami.
B. ecological degradation resulting from slash-and-burn farming techniques.
C. endless wars between neighboring Mayan city-states.
D. practices of interbreeding that led to a steep rise in congenital disorders.
Answer: B

Recent research on hominid species dating from the Middle Pliocene indicates there was (as of 2020):
A. a great amount of species diversity, or a single species that exhibited a lot of diversity.
B. very little species diversity during this period and very few hominids.
C. decreased species diversity due to a prolonged ice age followed by a severe drought.
D. decreased species diversity but increased numbers of hammerstones and flakes, indicating stone tool manufacture.
Answer: A

What is the approximate mean cranial capacity of Homo erectus?
A. under 650 cc
B. about 800 cc
C. just under 1000 cc
D. 1200 cc
Answer: C

When did Mohenjo-daro develop into a complex urban center?
A. during the Early Shang Dynasty, from 1,000 to 400 B.P.
B. during the Qin Dynasty, from 3,500 to 3,000 B.P.
C. during the Mature Harappan period, from 4,500 to 4,000 B.P.
D. during the Early Harappan period, from 8,500 to 6,000 B.P.
Answer:
```
Answer: **C**

### 87. [professional_law]

```
The following are multiple choice questions (with answers) about professional law.

A state legislature has recently enacted a statute making it a misdemeanor to curse or revile or use obscene or opprobrious language toward or in reference to a police officer perfonning his duties. A student at a state university organized a demonstration on campus to protest the war. The rally was attended by a group of 50 students who shouted anti-war messages at cars passing by. To show his contempt for the United States, the student sewed the American flag to the rear of his jeans. When a police officer saw the flag sown on the student's jeans, he approached and told him to remove the flag or he would be placed under arrest. The student became angered and shouted at the police officer, "Listen, you bastard, I'll wear this rag anywhere I please. " The student was subsequently placed under arrest and charged with violating the state statute. The student subsequently brings suit in state court challenging the constitutionality of the statute. The strongest constitutional argument for the student is that
A. the statute is void for vagueness under the Fourteenth Amendment's due process clause.
B. the statute is invalid because it violates the petitioner's freedom of speech under the First Amendment.
C. the statute is an abridgment of freedom of speech under the First Amendment because less restrictive means are available for achieving the same purpose.
D. the statute is overbroad and consequently invalid under the First and FourteenthAmendments.
Answer: D

A state has recently enacted a statute prohibiting the disposal of any nuclear wastes within the state. This law does not contravene or conflict with any federal statutes. A man operates a company in the state that is engaged in the disposal of nuclear wastes. Subsequent to the passage of the state statute, the man, not yet aware of the new law, entered into contracts with many out-of-state firms to dispose of their nuclear wastes in the state. On account of this new law, however, the man will be unable to perform these contracts. Assume that the man has standing to challenge this state law. Which of the following presents his strongest constitutional grounds to challenge the state law prohibiting the disposal of nuclear wastes within the state?
A. The commerce clause.
B. The equal protection clause of the Fourteenth Amendment.
C. The privileges and immunities clause of Article IV, Section 2. 
D. The contract clause.
Answer: A

Judge took judicial notice of some facts at the beginning of the trial. Which of the following is not an appropriate kind of fact for judicial notice?
A. Indisputable facts.
B. Facts that have been asserted by individual political organizations.
C. Facts recognized to be true by common knowledge.
D. Facts capable of scientific verification.
Answer: B

On October 1, 1980, a developer, owner of several hundred acres in a rural county, drafted a general development plan for the area. The duly recorded plan imposed elaborate limitations and restrictions upon the land in the plan, which was to be developed as a residential district. The restrictions were to extend to all persons acquiring any of the lots and to their heirs, assigns, and lessees. It was further provided that all subsequent owners would be charged with due notice of the restrictions. Among those restrictions in the general plan were the following:(22) A franchise right is created in a strip of land 10 feet in width along the rear of each lot for the use of public utility companies with right of ingress and egress. (23) No house or structure of any kind shall be built on the aforementioned strip of land running through the said blocks. In 2000, a retiree purchased one of the lots, built a house, and erected a fence in the rear of his property within the restricted area. In 2004, a teacher purchased a lot adjacent to the retiree's property and built a new house. Two years later, a librarian purchased the lot that adjoined the teacher's property. The three deeds to those properties each contained references to the deed book where the general plan was recorded. In 2008, the librarian began the construction of a seven-foot post-and-rail fence along the line dividing his lot with the teacher's, and along the center of the area subject to the franchise right. Although the teacher objected to its construction, the fence was completed. If the teacher seeks a mandatory injunction to compel removal of the librarian's fence, the court will most likely
A. grant relief, because the fence was in violation of the easement restriction. 
B. grant relief, because the encroachment of the fence violated the restriction in the original plan. 
C. deny relief, because the teacher failed to enforce the restriction against the retiree. 
D. deny relief, because the fence would not be construed as "a structure" within the terms of the restriction. 
Answer: B

A son owed a creditor $5,000. The son's father contacted the creditor and told him that he wanted to pay the son's debt. The father signed a document that stated the father would pay the son's debt at a rate of $500 a month for 10 months. The creditor made no written or oral commitment to forbear to sue the son to collect the $5,000 debt, and the father made no oral or written request for any such forbearance. For the next five months, the father made and the creditor accepted the $500 monthly payments as agreed. During that period, the creditor, in fact, did forbear to take any legal action against the son. However, the father then informed the creditor that he would make no further payments on the debt. Which of the following is the most persuasive argument that the father is liable to the creditor under the terms of their agreement?
A. The father's promise and the creditor's reliance thereon, if proved, gave rise to a valid claim by the creditor against the father based on the doctrine of promissory estoppel. 
B. Because it was foreseeable that the father's promise would induce the creditor to forbear taking any action against the son, such forbearance was, as a matter of law, a bargained-for consideration for the father's promise. 
C. The father's five payments to the creditor totaling $2,500 manifested a serious intent on the father's part to be contractually bound, and such manifestation is generally recognized as an effective substitute for consideration. 
D. By assuming the antecedent debt obligation that the son owed to the creditor, the father became a surety whose promise to the creditor was enforceable, since it was in writing and supported by adequate consideration. 
Answer: A

The Traditional Religion Advocacy Party (TRAP) has successfully lobbied the State of Texarkana for a statute named "Equal Rights for All Religions." The law requires broadcasters that air segments discussing Christianity to allow equal time for segments on Judaism, Islam, Hinduism, and Buddhism. The American Atheism Association (AAA) has put together a broadcast segment about atheism, rejecting all religions, but four of the five Texarkana stations have declined to air the segment because they believe it would violate the statute. AAA has filed suit against the state, challenging the constitutionality of the statute. What is the strongest argument AAA can make in support of striking down the Equal Rights for All Religions statute?
A. The First Amendment prohibits government interference with freedom of the press.
B. The fact that the statute involves only broadcast media and not print media is an Equal Protection violation.
C. The First Amendment prohibits government interference with the free exercise of religion.
D. The statute is not narrowly tailored to promote a compelling governmental interest.
Answer:
```
Answer: **D**

### 88. [nutrition]

```
The following are multiple choice questions (with answers) about nutrition.

In a cohort study, the risk ratio of developing diabetes was 0.86 when comparing consumers of tea (the exposed) to those who did not drink tea (the unexposed). Which one statement is correct (according to knowledge in 2020)?
A. The tea drinkers have lower risk of developing diabetes.
B. The tea drinkers have higher risk of developing diabetes.
C. Based on the information given we cannot tell if the observed difference in disease risk is the result of chance.
D. The risk ratio is close to the value one, so there is no difference in disease risk between the two groups.
Answer: C

Which of the following statements is correct (according to knowledge in 2020)?
A. Consumers with phenylketonuria must avoid the consumption of the sweetener aspartame
B. Consumers with phenylketonuria must avoid the consumption of the sweetener saccharin
C. Consumers with phenylketonuria must avoid the consumption of the sweetener sucralose
D. Consumers with phenylketonuria must avoid the consumption of the sweetener acesulfame K
Answer: A

Which of the following is the most plausible explanation for the protective effect of dietary fibre against cancer of the colon, as of 2020?
A. Propionic acid, formed during colonic fibre fermentation inhibits liver fatty acid synthesis
B. Butyric acid, formed during colonic fibre fermentation stimulates "silencing" of the SLC5A8 tumour suppressor gene
C. None of these options are correct
D. Butyric acid, formed during colonic fibre fermentation stimulates anti-oxidant defences in the colon
Answer: D

Which of the following statements about iodine is correct, as of 2020?
A. 50% of adults consume iodine at levels below the RNI
B. Dairy products are a poor source of iodine
C. The iodine content of organic milk is generally lower that the level in non-organic milk
D. UK dietary reference values recommend an increase in iodine intake in pregnancy
Answer: C

What is the first-line drug for patients with type 2 diabetes and obesity, as of 2020?
A. Acarbose
B. Metformin
C. Sulphonylureas
D. Insulin
Answer: B

Adaptive thermogenesis refers to:
A. A decrease in heat loss when exposed to cold
B. A decrease in non-shivering thermogenesis when exposed to cold
C. An increase in basal metabolic rate that is not fully explained by a change in body composition during chronic overfeeding
D. A decrease in spontaneous physical activity during chronic overfeeding.
Answer:
```
Answer: **C**

### 89. [human_sexuality]

```
The following are multiple choice questions (with answers) about human sexuality.

Morning sickness is typically a problem:
A. during the first trimester
B. during the second trimester
C. during the third trimester
D. all through the pregnancy
Answer: A

A woman who knows she has active herpes and untreated syphilis but continues to have sex without informing her partners of her condition has, in psychoanalytic terms:
A. a strong ego
B. a weak superego
C. a weak id
D. a strong superego
Answer: B

Women's ability to have multiple orgasms is primarily due to:
A. the fact that they do not have a refractory period.
B. the response of the inner layers of the vagina.
C. having alternating orgasms in different locations.
D. the G-Spot.
Answer: A

The nature of homosexual activities that occur during preadolescence include all but which of the following?
A. sexual intercourse
B. circle jerks
C. exhibitionism
D. touching each other's genitals
Answer: A

The most common disorder among men who seek sexual therapy is:
A. premature ejaculation
B. inhibited ejaculation
C. erectile disorder
D. ejaculatory disorder
Answer: C

In all of her relationships Helen is typically uncomfortable feeling close to another person or having that person feel close to her. According to attachment theory Helen would best be described as:
A. a secure lover
B. an avoidant lover
C. an anxious-ambivalent lover
D. a passionate lover
Answer:
```
Answer: **B**

### 90. [prehistory]

```
The following are multiple choice questions (with answers) about prehistory.

The great Mayan king Pacal built temples in the city of Palenque in order to:
A. satisfy the powerful Mayan astronomer priests.
B. display his generosity to the common people, since they were allowed to live in the temples.
C. frighten away enemies, in particular the Spaniards.
D. legitimize his kingship, since his father was not royal.
Answer: D

According to Timothy Pauketat, the evidence for social stratification and political power at Cahokia suggests:
A. a center of Mississippian civilization with conditions similar to the rise of early states.
B. the limitations of authority in a Native American society of egalitarian foragers.
C. a simple chiefdom or perhaps a complex chiefdom had evolved by A.D. 1500.
D. a center of Mississippian civilization with conditions similar to societies on the Northwest Coast of North America.
Answer: A

Researchers now believe that the decline of the Maya was caused chiefly by:
A. a cataclysm of some kind, such as an earthquake, volcano, or tsunami.
B. ecological degradation resulting from slash-and-burn farming techniques.
C. endless wars between neighboring Mayan city-states.
D. practices of interbreeding that led to a steep rise in congenital disorders.
Answer: B

Recent research on hominid species dating from the Middle Pliocene indicates there was (as of 2020):
A. a great amount of species diversity, or a single species that exhibited a lot of diversity.
B. very little species diversity during this period and very few hominids.
C. decreased species diversity due to a prolonged ice age followed by a severe drought.
D. decreased species diversity but increased numbers of hammerstones and flakes, indicating stone tool manufacture.
Answer: A

What is the approximate mean cranial capacity of Homo erectus?
A. under 650 cc
B. about 800 cc
C. just under 1000 cc
D. 1200 cc
Answer: C

In primates, the location of which of the following determines if it is a quadruped or biped?
A. the feet
B. the skull
C. the foramen magnum
D. the ulna
Answer:
```
Answer: **C**

### 91. [international_law]

```
The following are multiple choice questions (with answers) about international law.

What types of force does Article 2(4) of the UN Charter prohibit?
A. Article 2(4) encompasses only armed force
B. Article 2(4) encompasses all types of force, including sanctions
C. Article 2(4) encompasses all interference in the domestic affairs of States
D. Article 2(4) encompasses force directed only against a State's territorial integrity
Answer: A

What is the judge ad hoc?
A. If a party to a contentious case before the ICJ does not have a national sitting as judge, it is entitled to nominate someone as a judge solely for that case, with the title of judge ad hoc
B. Judge ad hoc is the member of the bench of the ICJ with a casting vote
C. Judge ad hoc is a surrogate judge, in case a judge is disqualified or passes away
D. Judge ad hoc is the judge that each party will always nominate in every contentious case
Answer: A

Would a reservation to the definition of torture in the ICCPR be acceptable in contemporary practice?
A. This is an acceptable reservation if the reserving country's legislation employs a different definition
B. This is an unacceptable reservation because it contravenes the object and purpose of the ICCPR
C. This is an unacceptable reservation because the definition of torture in the ICCPR is consistent with customary international law
D. This is an acceptable reservation because under general international law States have the right to enter reservations to treaties
Answer: B

When 'consent' can serve as a circumstance precluding the wrongfulness of a State conduct?
A. Consent can serve as a circumstance precluding the wrongfulness whenever it is given
B. Consent can never serve as a circumstance precluding wrongfulness
C. Consent can serve as a circumstance precluding wrongfulness, provided the consent is valid and to the extent that the conduct remains within the limits of the consent given
D. Consent can always serve as a circumstance precluding wrongfulness, no matter which organ of the State gives it
Answer: C

How the consent to be bound of a State may be expressed?
A. The consent of a State to be bound is expressed only by ratification
B. The consent of a state to be bound by a treaty may be expressed by signature, ratification, acceptance, approval or accession
C. The consent of a State to be bound is expressed by signature
D. The consent of a State to be bound is expressed by whatever means they choose
Answer: B

What is the meaning of collective security?
A. The right to self-defence by more than one nation acting in concert
B. The right of one's allies to defend the victim State
C. The authorisation of armed force by the UN Security Council
D. The authorisation of peacekeeping missions by the UN General Assembly
Answer:
```
Answer: **C**

### 92. [miscellaneous]

```
The following are multiple choice questions (with answers) about miscellaneous.

How many axles does a standard automobile have?
A. one
B. two
C. four
D. eight
Answer: B

What place is named in the title of the 1979 live album by rock legends Cheap Trick?
A. Budapest
B. Budokan
C. Bhutan
D. Britain
Answer: B

Who is the shortest man to ever win an NBA slam dunk competition?
A. Anthony 'Spud' Webb
B. Michael 'Air' Jordan
C. Tyrone 'Muggsy' Bogues
D. Julius 'Dr J' Erving
Answer: A

What is produced during photosynthesis?
A. hydrogen
B. nylon
C. oxygen
D. light
Answer: C

Which of these songs was a Top 10 hit for the rock band The Police?
A. 'Radio Ga-Ga'
B. 'Ob-la-di Ob-la-da'
C. 'De Do Do Do De Da Da Da'
D. 'In-a-Gadda-Da-Vida'
Answer: C

In the comic strip 'Peanuts' what is Schroeder known for doing?
A. Dancing
B. playing football
C. playing the piano
D. Flying an imaginary plane
Answer:
```
Answer: **C**

### 93. [professional_law]

```
The following are multiple choice questions (with answers) about professional law.

A state legislature has recently enacted a statute making it a misdemeanor to curse or revile or use obscene or opprobrious language toward or in reference to a police officer perfonning his duties. A student at a state university organized a demonstration on campus to protest the war. The rally was attended by a group of 50 students who shouted anti-war messages at cars passing by. To show his contempt for the United States, the student sewed the American flag to the rear of his jeans. When a police officer saw the flag sown on the student's jeans, he approached and told him to remove the flag or he would be placed under arrest. The student became angered and shouted at the police officer, "Listen, you bastard, I'll wear this rag anywhere I please. " The student was subsequently placed under arrest and charged with violating the state statute. The student subsequently brings suit in state court challenging the constitutionality of the statute. The strongest constitutional argument for the student is that
A. the statute is void for vagueness under the Fourteenth Amendment's due process clause.
B. the statute is invalid because it violates the petitioner's freedom of speech under the First Amendment.
C. the statute is an abridgment of freedom of speech under the First Amendment because less restrictive means are available for achieving the same purpose.
D. the statute is overbroad and consequently invalid under the First and FourteenthAmendments.
Answer: D

A state has recently enacted a statute prohibiting the disposal of any nuclear wastes within the state. This law does not contravene or conflict with any federal statutes. A man operates a company in the state that is engaged in the disposal of nuclear wastes. Subsequent to the passage of the state statute, the man, not yet aware of the new law, entered into contracts with many out-of-state firms to dispose of their nuclear wastes in the state. On account of this new law, however, the man will be unable to perform these contracts. Assume that the man has standing to challenge this state law. Which of the following presents his strongest constitutional grounds to challenge the state law prohibiting the disposal of nuclear wastes within the state?
A. The commerce clause.
B. The equal protection clause of the Fourteenth Amendment.
C. The privileges and immunities clause of Article IV, Section 2. 
D. The contract clause.
Answer: A

Judge took judicial notice of some facts at the beginning of the trial. Which of the following is not an appropriate kind of fact for judicial notice?
A. Indisputable facts.
B. Facts that have been asserted by individual political organizations.
C. Facts recognized to be true by common knowledge.
D. Facts capable of scientific verification.
Answer: B

On October 1, 1980, a developer, owner of several hundred acres in a rural county, drafted a general development plan for the area. The duly recorded plan imposed elaborate limitations and restrictions upon the land in the plan, which was to be developed as a residential district. The restrictions were to extend to all persons acquiring any of the lots and to their heirs, assigns, and lessees. It was further provided that all subsequent owners would be charged with due notice of the restrictions. Among those restrictions in the general plan were the following:(22) A franchise right is created in a strip of land 10 feet in width along the rear of each lot for the use of public utility companies with right of ingress and egress. (23) No house or structure of any kind shall be built on the aforementioned strip of land running through the said blocks. In 2000, a retiree purchased one of the lots, built a house, and erected a fence in the rear of his property within the restricted area. In 2004, a teacher purchased a lot adjacent to the retiree's property and built a new house. Two years later, a librarian purchased the lot that adjoined the teacher's property. The three deeds to those properties each contained references to the deed book where the general plan was recorded. In 2008, the librarian began the construction of a seven-foot post-and-rail fence along the line dividing his lot with the teacher's, and along the center of the area subject to the franchise right. Although the teacher objected to its construction, the fence was completed. If the teacher seeks a mandatory injunction to compel removal of the librarian's fence, the court will most likely
A. grant relief, because the fence was in violation of the easement restriction. 
B. grant relief, because the encroachment of the fence violated the restriction in the original plan. 
C. deny relief, because the teacher failed to enforce the restriction against the retiree. 
D. deny relief, because the fence would not be construed as "a structure" within the terms of the restriction. 
Answer: B

A son owed a creditor $5,000. The son's father contacted the creditor and told him that he wanted to pay the son's debt. The father signed a document that stated the father would pay the son's debt at a rate of $500 a month for 10 months. The creditor made no written or oral commitment to forbear to sue the son to collect the $5,000 debt, and the father made no oral or written request for any such forbearance. For the next five months, the father made and the creditor accepted the $500 monthly payments as agreed. During that period, the creditor, in fact, did forbear to take any legal action against the son. However, the father then informed the creditor that he would make no further payments on the debt. Which of the following is the most persuasive argument that the father is liable to the creditor under the terms of their agreement?
A. The father's promise and the creditor's reliance thereon, if proved, gave rise to a valid claim by the creditor against the father based on the doctrine of promissory estoppel. 
B. Because it was foreseeable that the father's promise would induce the creditor to forbear taking any action against the son, such forbearance was, as a matter of law, a bargained-for consideration for the father's promise. 
C. The father's five payments to the creditor totaling $2,500 manifested a serious intent on the father's part to be contractually bound, and such manifestation is generally recognized as an effective substitute for consideration. 
D. By assuming the antecedent debt obligation that the son owed to the creditor, the father became a surety whose promise to the creditor was enforceable, since it was in writing and supported by adequate consideration. 
Answer: A

A writer's latest novel created a debate among critics regarding whether the sexual references and the graphic practices in the book made it obscene. While the publisher was readying the book for distribution, the head of a state commission for morality contacted her to tell her that the book was obscene and that she should collect all copies and turn them over for seizure by the government. He also advised that the commission was putting the book on a list of obscene works that retailers should not handle. She responded that dozens of retailers already had the book for an opening day promotion. The commission seized all of the books from the retailers and destroyed them, without any due process procedural safeguards to the author or the publisher. The author and publisher appealed the case and eventually received a writ of certiorari issued by the U.S. Supreme Court. Will the Court uphold the procedure that was used to declare the book obscene and to restrain its distribution?
A. Yes, if the authorities determine the book to be obscene, then they may seize and prevent its prospective distribution for the protection of the public.
B. Yes, the determination of obscenity can be made prior to publication because the local authorities have a right to determine the local "community morals" standard.
C. No, the state must first give the publisher and author a preliminary hearing before a magistrate to get a determination if the book is obscene.
D. No, the state cannot exercise prior restraint over published material because that has a chilling effect on free speech; a full judicial determination, with a criminal trial, is required before there can be a finding of a work being obscene.
Answer:
```
Answer: **D**

### 94. [high_school_european_history]

```
The following are multiple choice questions (with answers) about high school european history.

This question refers to the following information.
The following excerpt is from a pamphlet.
You will do me the justice to remember, that I have always strenuously supported the Right of every man to his own opinion, however different that opinion might be to mine. He who denies to another this right, makes a slave of himself to his present opinion, because he precludes himself the right of changing it.
The most formidable weapon against errors of every kind is Reason. I have never used any other, and I trust I never shall.
The circumstance that has now taken place in France of the total abolition of the whole national order of priesthood, and of everything appertaining to compulsive systems of religion, and compulsive articles of faith, has not only precipitated my intention, but rendered a work of this kind exceedingly necessary, lest in the general wreck of superstition, of false systems of government, and false theology, we lose sight of morality, of humanity, and of the theology that is true.
I believe in one God, and no more; and I hope for happiness beyond this life.
I believe in the equality of man; and I believe that religious duties consist in doing justice, loving mercy, and endeavoring to make our fellow-creatures happy.
I do not believe in the creed professed by the Jewish church, by the Roman church, by the Greek church, by the Turkish church, by the Protestant church, nor by any church that I know of. My own mind is my own church.
All national institutions of churches, whether Jewish, Christian or Turkish, appear to me no other than human inventions, set up to terrify and enslave mankind, and monopolize power and profit.
I do not mean by this declaration to condemn those who believe otherwise; they have the same right to their belief as I have to mine.
—Thomas Paine, The Age of Reason, 1794–1795
Which of the following Enlightenment philosophes designed a system of checks and balances for government to avoid abuses of power?
A. Jean Jacques Rousseau
B. Baron Montesquieu
C. Mary Wollstonecraft
D. Adam Smith
Answer: B

This question refers to the following information.
Read the following excerpt.
The revolutionary seed had penetrated into every country and spread more or less. It was greatly developed under the régime of the military despotism of Bonaparte. His conquests displaced a number of laws, institutions, and customs; broke through bonds sacred among all nations, strong enough to resist time itself; which is more than can be said of certain benefits conferred by these innovators.
The monarchs will fulfil the duties imposed upon them by Him who, by entrusting them with power, has charged them to watch over the maintenance of justice, and the rights of all, to avoid the paths of error, and tread firmly in the way of truth. Placed beyond the passions which agitate society, it is in days of trial chiefly that they are called upon to despoil realities of their false appearances, and to show themselves as they are, fathers invested with the authority belonging by right to the heads of families, to prove that, in days of mourning, they know how to be just, wise, and therefore strong, and that they will not abandon the people whom they ought to govern to be the sport of factions, to error and its consequences, which must involve the loss of society.
Union between the monarchs is the basis of the policy which must now be followed to save society from total ruin. . . .
Let them not confound concessions made to parties with the good they ought to do for their people, in modifying, according to their recognized needs, such branches of the administration as require it.
Let them be just, but strong; beneficent, but strict.
Let them maintain religious principles in all their purity, and not allow the faith to be attacked and morality interpreted according to the social contract or the visions of foolish sectarians.
Let them suppress Secret Societies; that gangrene of society.
—Klemens von Metternich, Political Confession of Faith, 1820
Which of the following was the greatest cause of the fears expressed by Metternich in the document above?
A. The ideas of personal liberty and nationalism conceived during the Enlightenment resulted in radical revolutions that could spread throughout Europe.
B. The conquest of Europe by Napoleon led to the creation of new factions and shifted the European balance of power.
C. The power of monarchs had grown to the point where it needed to be checked by other powers within each nation or domination of civilians would occur.
D. The rising and falling economic cycle of the newly emerging capitalist economy could lead to civilian unrest that must be suppressed.
Answer: A

This question refers to the following information.
In Russia there was nothing going on well, and [Souvarine] was in despair over the news he had received. His old companions were all turning to the politicians; the famous Nihilists who made Europe tremble-sons of village priests, of the lower middle class, of tradesmen-could not rise above the idea of national liberation, and seemed to believe that the world would be delivered-when they had killed their despot&…
"Foolery! They'll never get out of it with their foolery."
Then, lowering his voice still more, in a few bitter words he described his old dream of fraternity. He had renounced his rank and his fortune; he had gone among workmen, only in the hope of seeing at last the foundation of a new society of labour in common. All the sous in his pockets had long gone to the urchins of the settlement; he had been as tender as a brother with the colliers, smiling at their suspicion, winning them over by his quiet workmanlike ways and his dislike of chattering. But decidedly the fusion had not taken place.
His voice changed, his eyes grew bright, he fixed them on étienne, directly addressing him:
"Now, do you understand that? These hatworkers at Marseilles who have won the great lottery prize of a hundred thousand francs have gone off at once and invested it, declaring that they are going to live without doing anything! Yes, that is your idea, all of you French workmen; you want to unearth a treasure in order to devour it alone afterwards in some lazy, selfish corner. You may cry out as much as you like against the rich, you haven't got courage enough to give back to the poor the money that luck brings you. You will never be worthy of happiness as long as you own anything, and your hatred of the bourgeois proceeds solely from an angry desire to be bourgeois yourselves in their place."
émile Zola, French writer, Germinal, 1885
The passage displays the direct concern for the welfare of the working classes that was typically a part of which movement?
A. Capitalist
B. Scientific
C. Communist
D. Existentialist
Answer: C

This question refers to the following information.
The excerpts below are from the Navigation Acts of 1651.
[A]fter the first day of December, one thousand six hundred fifty and one, and from thence forwards, no goods or commodities whatsoever of the growth, production or manufacture of Asia, Africa or America, or of any part thereof; or of any islands belonging to them, or which are described or laid down in the usual maps or cards of those places, as well of the English plantations as others, shall be imported or brought into this Commonwealth of England, or into Ireland, or any other lands, islands, plantations, or territories to this Commonwealth belonging, or in their possession, in any other ship or ships, vessel or vessels whatsoever, but only in such as do truly and without fraud belong only to the people of this Commonwealth, or the plantations thereof, as the proprietors or right owners thereof; and whereof the master and mariners are also of the people of this Commonwealth, under the penalty of the forfeiture and loss of all the goods that shall be imported contrary to this act, , , ,
[N]o goods or commodities of the growth, production, or manufacture of Europe, or of any part thereof, shall after the first day of December, one thousand six hundred fifty and one, be imported or brought into this Commonwealth of England, or any other lands or territories to this Commonwealth belonging, or in their possession, in any ship or ships, vessel or vessels whatsoever, but in such as do truly and without fraud belong only to the people of this Commonwealth, and in no other, except only such foreign ships and vessels as do truly and properly belong to the people of that country or place, of which the said goods are the growth, production or manufacture.
Which of the following best describes the outcome of the Navigation Acts of 1651?
A. They served as a catalyst for the growth of English shipping and overseas trade, but did little to limit the prospects of the Dutch in the seventeenth century.
B. They brought about almost immediate hardships for the Dutch economy as their dominance of overseas trade quickly ended.
C. They were rescinded during the restoration of the Stuarts as they sought normal diplomatic relations with the Dutch so not as to need Parliament's financial support for war.
D. They led to nearly a century of recurrent war between England and the Netherlands, which would not end until after American independence.
Answer: A

This question refers to the following information.
Albeit the king's Majesty justly and rightfully is and ought to be the supreme head of the Church of England, and so is recognized by the clergy of this realm in their convocations, yet nevertheless, for corroboration and confirmation thereof, and for increase of virtue in Christ's religion within this realm of England, and to repress and extirpate all errors, heresies, and other enormities and abuses heretofore used in the same, be it enacted, by authority of this present Parliament, that the king, our sovereign lord, his heirs and successors, kings of this realm, shall be taken, accepted, and reputed the only supreme head in earth of the Church of England, called Anglicans Ecclesia; and shall have and enjoy, annexed and united to the imperial crown of this realm, as well the title and style thereof, as all honors, dignities, preeminences, jurisdictions, privileges, authorities, immunities, profits, and commodities to the said dignity of the supreme head of the same Church belonging and appertaining; and that our said sovereign lord, his heirs and successors, kings of this realm, shall have full power and authority from time to time to visit, repress, redress, record, order, correct, restrain, and amend all such errors, heresies, abuses, offenses, contempts, and enormities, whatsoever they be, which by any manner of spiritual authority or jurisdiction ought or may lawfully be reformed, repressed, ordered, redressed, corrected, restrained, or amended, most to the pleasure of Almighty God, the increase of virtue in Christ's religion, and for the conservation of the peace, unity, and tranquility of this realm; any usage, foreign land, foreign authority, prescription, or any other thing or things to the contrary hereof notwithstanding.
English Parliament, Act of Supremacy, 1534
From the passage, one may infer that the English Parliament wished to argue that the Act of Supremacy would
A. give the English king a new position of authority
B. give the position of head of the Church of England to Henry VIII alone and exclude his heirs
C. establish Calvinism as the one true theology in England
D. end various forms of corruption plaguing the Church in England
Answer: D

This question refers to the following information.
"But you, my dear Pangloss," said Candide, "how can it be that I behold you again?"
"It is true," said Pangloss, "that you saw me hanged&….A surgeon purchased my body, carried home, and dissected me. He began with making a crucial incision on me from the navel to the clavicula. One could not have been worse hanged than I was. The executioner of the Holy Inquisition was a sub-deacon, and knew how to burn people marvellously well, but he was not accustomed to hanging. The cord was wet and did not slip properly, and besides it was badly tied; in short, I still drew my breath, when the crucial incision made me give such a frightful scream that my surgeon fell flat upon his back&…[At length he] sewed up my wounds; his wife even nursed me. I was upon my legs at the end of fifteen days&….
One day I took it into my head to step into a mosque, where I saw an old Iman and a very pretty young devotee who was saying her paternosters&….She dropped her bouquet; I picked it up, and presented it to her with a profound reverence. I was so long in delivering it that the Iman began to get angry, and seeing that I was a Christian he called out for help. They carried me before the cadi, who ordered me a hundred lashes on the soles of the feet and sent me to the galleys. I was chained to the very same galley and the same bench as the young Baron. On board this galley there were four young men from Marseilles, five Neapolitan priests, and two monks from Corfu, who told us similar adventures happened daily. The Baron maintained that he had suffered greater injustice than I&….We were continually disputing, and received twenty lashes with a bull's pizzle when the concatenation of universal events brought you to our galley, and you were good enough to ransom us."
"Well, my dear Pangloss," said Candide to him, "when you had been hanged, dissected, whipped, and were tugging at the oar, did you always think that everything happens for the best?"
"I am still of my first opinion," answered Pangloss, "for I am a philosopher and I cannot retract, especially as Leibnitz could never be wrong; and besides, the pre-established harmony is the finest thing in the world, and so is his plenum and materia subtilis."
Voltaire, French Enlightenment writer, Candide, 1759
The themes of the passage and the mode in which Pangloss tells them show the influence of
A. Medievalism
B. Empiricism
C. Rationalism
D. Romanticism
Answer:
```
Answer: **C**

### 95. [college_biology]

```
The following are multiple choice questions (with answers) about college biology.

Which of the following represents an accurate statement concerning arthropods?
A. They possess an exoskeleton composed primarily of peptidoglycan.
B. They possess an open circulatory system with a dorsal heart.
C. They are members of a biologically unsuccessful phylum incapable of exploiting diverse habitats and nutrition sources.
D. They lack paired, jointed appendages.
Answer: B

In a given population, 1 out of every 400 people has a cancer caused by a completely recessive allele, b. Assuming the population is in Hardy-Weinberg equilibrium, which of the following is the expected proportion of individuals who carry the b allele but are not expected to develop the cancer?
A. 1/400
B. 19/400
C. 20/400
D. 38/400
Answer: D

The presence of homologous structures in two different organisms, such as the humerus in the front limb of a human and a bird, indicates that
A. the human and bird are polyphyletic species
B. a human's and bird's evolution is convergent
C. the human and bird belong to a clade
D. the human and bird developed by analogy
Answer: C

According to the pressure-flow model of movement of phloem contents, photosynthate movement from source to sink is driven by
A. an ATP-dependent pressure-flow pump
B. a water-pressure potential gradient
C. transpiration
D. apoplastic diffusion
Answer: B

Which of the following contain DNA sequences required for the segregation of chromosomes in mitosis and meiosis?
A. Telomeres
B. Centromeres
C. Nucleosomes
D. Spliceosomes
Answer: B

Which of the following is the symplastic pathway for the movement of sucrose from the site of photosynthesis in mesophyll cells into the phloem?
A. Fibers, phloem parenchyma, companion cell, sieve tube
B. Phloem parenchyma, fibers, bundle sheath, tracheids
C. Companion cells, phloem parenchyma, fibers, sieve tube
D. Bundle sheath, phloem parenchyma, companion cell, sieve tube
Answer:
```
Answer: **D**

### 96. [professional_medicine]

```
The following are multiple choice questions (with answers) about professional medicine.

A 42-year-old man comes to the office for preoperative evaluation prior to undergoing adrenalectomy scheduled in 2 weeks. One month ago, he received care in the emergency department for pain over his right flank following a motor vehicle collision. At that time, blood pressure was 160/100 mm Hg and CT scan of the abdomen showed an incidental 10-cm left adrenal mass. Results of laboratory studies, including complete blood count, serum electrolyte concentrations, and liver function tests, were within the reference ranges. The patient otherwise had been healthy and had never been told that he had elevated blood pressure. He takes no medications. A follow-up visit in the office 2 weeks ago disclosed elevated urinary normetanephrine and metanephrine and plasma aldosterone concentrations. The patient was referred to a surgeon, who recommended the adrenalectomy. Today, vital signs are temperature 36.6°C (97.9°F), pulse 100/min, respirations 14/min, and blood pressure 170/95 mm Hg. Physical examination discloses no significant findings. Initial preoperative preparation should include treatment with which of the following?
A. Labetalol
B. A loading dose of potassium chloride
C. Nifedipine
D. Phenoxybenzamine
Answer: D

A 36-year-old male presents to the office with a 3-week history of low back pain. He denies any recent trauma but says that he climbs in and out of his truck numerous times a day for his job. Examination of the patient in the prone position reveals a deep sacral sulcus on the left, a posterior inferior lateral angle on the right, and a lumbosacral junction that springs freely on compression. The most likely diagnosis is
A. left-on-left sacral torsion
B. left-on-right sacral torsion
C. right unilateral sacral flexion
D. right-on-right sacral torsion
Answer: D

A previously healthy 32-year-old woman comes to the physician 8 months after her husband was killed in a car crash. Since that time, she has had a decreased appetite and difficulty falling asleep. She states that she is often sad and cries frequently. She has been rechecking the door lock five times before leaving her house and has to count exactly five pieces of toilet paper before she uses it. She says that she has always been a perfectionist but these urges and rituals are new. Pharmacotherapy should be targeted to which of the following neurotransmitters?
A. Dopamine
B. Glutamate
C. Norepinephrine
D. Serotonin
Answer: D

A 44-year-old man comes to the office because of a 3-day history of sore throat, nonproductive cough, runny nose, and frontal headache. He says the headache is worse in the morning and ibuprofen does provide some relief. He has not had shortness of breath. Medical history is unremarkable. He takes no medications other than the ibuprofen for pain. Vital signs are temperature 37.4°C (99.4°F), pulse 88/min, respirations 18/min, and blood pressure 120/84 mm Hg. Examination of the nares shows erythematous mucous membranes. Examination of the throat shows erythema and follicular lymphoid hyperplasia on the posterior oropharynx. There is no palpable cervical adenopathy. Lungs are clear to auscultation. Which of the following is the most likely cause of this patient's symptoms?
A. Allergic rhinitis
B. Epstein-Barr virus
C. Mycoplasma pneumoniae
D. Rhinovirus
Answer: D

A 22-year-old male marathon runner presents to the office with the complaint of right-sided rib pain when he runs long distances. Physical examination reveals normal heart and lung findings and an exhalation dysfunction at ribs 4-5 on the right. Which of the following muscles or muscle groups will be most useful in correcting this dysfunction utilizing a direct method?
A. anterior scalene
B. latissimus dorsi
C. pectoralis minor
D. quadratus lumborum
Answer: C

A 45-year-old man comes to the physician because of right shoulder pain that began after he chopped wood 2 days ago. Examination of the right upper extremity shows no obvious bone deformities or point tenderness. The pain is reproduced when the patient is asked to externally rotate the shoulder against resistance; there is no weakness. In addition to the teres minor, inflammation of which of the following tendons is most likely in this patient?
A. Infraspinatus
B. Pectoralis
C. Subscapularis
D. Supraspinatus
Answer:
```
Answer: **A**

### 97. [formal_logic]

```
The following are multiple choice questions (with answers) about formal logic.

Select the best translation into predicate logic: No people drive on Mars.
A. ~Pd
B. (∀x)(Px ∨ ~Dx)
C. (∀x)(Px ⊃ ~Dx)
D. ~Dp
Answer: C

Select the best translation into predicate logic.George borrows Hector's lawnmower. (g: George; h: Hector; l: Hector's lawnmower; Bxyx: x borrows y from z)
A. Blgh
B. Bhlg
C. Bglh
D. Bghl
Answer: C

Select the best English interpretation of the given arguments in predicate logic.
Dm
(∀x)(Wx ⊃ ~Dx)
(∀x)Wx ∨ Ag	/ (∃x)Ax
A. Marina is a dancer. Some weaklings are not dancers. Either everything is a weakling or Georgia plays volleyball. So something plays volleyball.
B. Marina is a dancer. No weakling is a dancer. Everything is either a weakling or plays volleyball. So something plays volleyball.
C. Marina is a dancer. Some weaklings are not dancers. Everything is either a weakling or plays volleyball. So something plays volleyball.
D. Marina is a dancer. No weakling is a dancer. Either everything is a weakling or Georgia plays volleyball. So something plays volleyball.
Answer: D

Construct a complete truth table for the following pairs of propositions. Then, using the truth tables, determine whether the statements are logically equivalent or contradictory. If neither, determine whether they are consistent or inconsistent. Justify your answers.
E ⊃ (F · E) and ~E · F
A. Logically equivalent
B. Contradictory
C. Neither logically equivalent nor contradictory, but consistent
D. Inconsistent
Answer: C

Which of the given formulas of PL is the best symbolization of the following sentence?
Turtles live long lives and are happy creatures, unless they are injured.
A. (L • H) ≡ I
B. (L • H) ∨ I
C. L • (H ∨ I)
D. L • (H ⊃ R)
Answer: B

Use the following key to translate the given formula of PL to natural, English sentences.
A: Marina reads a Percy Jackson book.
B: Izzy plays Minecraft.
C: Emily stops working.
D: Russell makes dinner.
E: Ashleigh stops by.
~C ∨ D
A. If Emily doesn't stop working then Russell makes dinner.
B. Emily stops working unless Russell makes dinner.
C. Emily stops working unless Russell doesn't make dinner.
D. Emily doesn't stop working unless Russell makes dinner.
Answer:
```
Answer: **D**

### 98. [high_school_chemistry]

```
The following are multiple choice questions (with answers) about high school chemistry.

Which of the following is considered an acid anhydride?
A. HCl
B. H2SO3
C. SO2
D. Al(NO3)3
Answer: C

Which of the following is expected to be a polar molecule?
A. PCl4F
B. BF3
C. CO2
D. Si(CH3)4
Answer: A

From the solubility rules, which of the following is true?
A. All chlorides, bromides, and iodides are soluble
B. All sulfates are soluble
C. All hydroxides are soluble
D. All ammonium-containing compounds are soluble
Answer: D

A new compound is synthesized and found to be a monoprotic acid with a molar mass of 248 g/mol. When 0.0050 mol of this acid are dissolved in 0.500 L of water, the pH is measured as 3.89. What is the pKa of this acid?
A. 3.89
B. 7.78
C. 5.78
D. 2.33
Answer: C

A solution contains 2.00 mole of acetic acid, CH3COOH, and 1.00 mole of calcium acetate, Ca(CH3COO)2. The solution is able to resist the addition of a small amount of strong acid or strong base with only minor changes in the pH of the solution. Larger quantities of strong acid or strong base can cause a significant change in pH. How many moles of nitric acid, HNO3, may be added before the pH begins to change significantly?
A. 0.500 mole
B. 1.00 mole
C. 2.00 mole
D. 3.00 mole
Answer: C

The weight percent of sodium hydroxide dissolved in water is 50%. What is the mole fraction of sodium hydroxide?
A. 31.00%
B. 0.164
C. 0.311
D. 0.5
Answer:
```
Answer: **C**

### 99. [high_school_physics]

```
The following are multiple choice questions (with answers) about high school physics.

Which of the following conditions will ensure that angular momentum is conserved? I. Conservation of linear momentum II. Zero net external force III. Zero net external torque
A. I and II only
B. I and III only
C. II and III only
D. III only
Answer: D

A pipe full of air is closed at one end. A standing wave is produced in the pipe, causing the pipe to sound a note. Which of the following is a correct statement about the wave’s properties at the closed end of the pipe?
A. The pressure is at a node, but the particle displacement is at an antinode.
B. The pressure is at an antinode, but the particle displacement is at a node.
C. The pressure and the particle displacement are both at nodes.
D. The pressure and the particle displacement are both at antinodes.
Answer: B

A photocell of work function ϕ = 2eV is connected to a resistor in series. Light of frequency f = 1 × 10^15 Hz hits a metal plate of the photocell. If the power of the light is P = 100 W, what is the current through the resistor?
A. 2:00 AM
B. 6:00 AM
C. 12:00 AM
D. 24 A
Answer: D

A microwave oven is connected to an outlet, 120 V, and draws a current of 2 amps. At what rate is energy being used by the microwave oven?
A. 10 W
B. 30 W
C. 60 W
D. 240 W
Answer: D

A point charge, Q = +1 mC, is fixed at the origin. How much work is required to move a charge, Q = +8 µC, from the point (0, 4 meters) to the point (3 meters, 0)?
A. 3.5 J
B. 6.0 J
C. 22.5 J
D. 40 J
Answer: B

In a certain region, the electric field varies with the radius away from origin by the equation Er = –6r^2 + 4r + 3, where r is given in meters and E in N/C. The potential difference between the origin and the point (3, 4) is
A. –165 V
B. –120 V
C. 64 V
D. 185 V
Answer:
```
Answer: **D**
