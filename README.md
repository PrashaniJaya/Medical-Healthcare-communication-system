# Medical-Healthcare-communication-system
The medical communication system first starts with creation of multiagents to dynamically process the query. 
The First agent is RAG and in the repo there are 4 types
  1. Rag agent with medical dialog from hugging face
  2. Rag agent with URLS directly inserted to RAG
  3. Rag with a webscraper
  4. Rag with both a medical dialog and URLs

Each of these agents provided the following answers:

Test 001 for the questions with simple RAG:
Query: What is the morning after pill made of?

- The morning-after pill is a form of emergency contraception used to reduce the risk of pregnancy after unprotected sex or contraceptive failure (e.g., condom break). (score: 0.64)
Morning-after pill guide (Source: https://www.drugs.com/mtm/morning-after.html) (score: 0.64)
1. Morning-After Pill (Emergency Contraception) (score: 0.64)
==================================================

Query: Are there any side effects of taking emergency contraception?

1. Morning-After Pill (Emergency Contraception) (score: 0.63)
- Emergency contraceptives can cause hormonal fluctuations, which may trigger or worsen migraines in susceptible individuals. (score: 0.58)
Emergency contraception info (Source: https://www.drugs.com/condition/postcoital-contraception.html) (score: 0.57)
==================================================

Query: How long after sex can the morning after pill work?

Morning-after pill guide (Source: https://www.drugs.com/mtm/morning-after.html) (score: 0.60)
1. Morning-After Pill (Emergency Contraception) (score: 0.57)
- The morning-after pill is a form of emergency contraception used to reduce the risk of pregnancy after unprotected sex or contraceptive failure (e.g., condom break). (score: 0.56)

 Query: What are the side effects of Unwanted 72?
Unwanted 72, an emergency contraceptive pill, can cause side effects such as nausea, vomiting, headache, abdominal pain, irregular bleeding or spotting, fatigue, and breast tenderness. These effects are generally temporary and should resolve within a few days. If taken without a valid reason, it may lead to changes in the menstrual cycle, including delays or altered bleeding patterns. It's important to note that Unwanted 72 should not be used as a regular contraceptive method.

#Test 002 With meddialog
Query: Is it safe to take the morning after pill multiple times in one month?
Taking the morning-after pill multiple times in one month is not recommended. While it is safe to use occasionally as an emergency contraceptive, relying on it frequently can lead to side effects such as irregular periods and nausea. It is designed for emergency use and should not replace regular contraceptive methods. For ongoing contraception, it's best to consult a healthcare provider to explore more effective and sustainable options.

==================================================
 Query: How effective is the i-pill compared to regular contraceptive pills?
The i-pill, an emergency contraceptive, is designed to prevent pregnancy after unprotected sex and is effective when taken within 72 hours. In contrast, regular contraceptive pills are taken daily to prevent ovulation and are not intended for emergency use. While the i-pill is effective for its purpose, it should not be used as a regular contraceptive method due to potential side effects and irregularities in menstrual cycles. For ongoing contraception, regular birth control pills are recommended.

This information is for educational purposes only and not a substitute for professional medical advice.

Test 003 With only URLS
Query: What are the side effects of the morning after pill?
The morning-after pill can cause several side effects, which may include:

- Nausea and vomiting
- Fatigue
- Headaches
- Dizziness
- Breast tenderness
- Changes in menstrual bleeding (such as heavier or lighter periods)

These side effects are generally mild and temporary. If you experience severe symptoms or if your period is more than a week late after taking the pill, it's advisable to consult a healthcare provider.

This information is for educational purposes only and not a substitute for professional medical advice.

Sources:
- https://www.drugs.com/mtm/morning-after.html
- https://www.drugs.com/condition/postcoital-contraception.html
- https://www.plannedparenthood.org/learn/morning-after-pill-emergency-contraception
==================================================

 Query: Is it safe to use emergency contraception if I have PCOD?
Yes, it is generally safe to use emergency contraception if you have Polycystic Ovary Syndrome (PCOS). Emergency contraception, such as the morning-after pill, works by preventing ovulation or fertilization and is not contraindicated for individuals with PCOS. However, it is important to consult with a healthcare provider to discuss your specific health situation and any potential interactions with other medications you may be taking.

This information is for educational purposes only and not a substitute for professional medical advice.

Sources:
- https://www.drugs.com/condition/postcoital-contraception.html
- https://www.plannedparenthood.org/learn/morning-after-pill-emergency-contraception
- https://www.drugs.com/mtm/morning-after.html
==================================================

 Query: Difference between regular contraceptive pills and i-pill?
Regular contraceptive pills are designed for ongoing use to prevent pregnancy by regulating hormones, while the i-pill (or morning-after pill) is a form of emergency contraception intended for use after unprotected intercourse or contraceptive failure. Regular pills are taken daily, whereas the i-pill is taken as soon as possible after the incident, ideally within 72 hours, but can be effective up to 5 days later. The i-pill contains a higher dose of hormones compared to regular contraceptive pills and is not meant for regular use.

This information is for educational purposes only and not a substitute for professional medical advice.

Sources:
- https://www.drugs.com/condition/postcoital-contraception.html
- https://www.drugs.com/mtm/morning-after.html
- https://www.plannedparenthood.org/learn/morning-after-pill-emergency-contraception
With web scraping
Query: What are the side effects of the morning after pill?
The morning after pill, also known as emergency contraception, can have several side effects. Common side effects include nausea, fatigue, headache, dizziness, breast tenderness, and changes in menstrual bleeding. Some women may experience a delay in their next period or heavier or lighter bleeding than usual. 

It's important to note that while these side effects can occur, they are generally temporary and resolve on their own. If you have concerns about side effects or if symptoms persist, it's advisable to consult a healthcare professional.

==================================================

 Query: How long after sex can I take emergency contraception?
Emergency contraception can be taken after unprotected sex, and its effectiveness depends on the type used. There are two main types of emergency contraception pills: 

1. Levonorgestrel (Plan B One-Step): This can be taken up to 72 hours (3 days) after unprotected intercourse, but it is most effective when taken as soon as possible.

2. Ulipristal Acetate (Ella): This can be taken up to 120 hours (5 days) after unprotected sex and is effective throughout that entire period.

It's important to note that emergency contraception is not intended for regular use and should not replace regular contraceptive methods.

==================================================

 Query: Is the morning after pill safe for people with PCOD?
The morning-after pill is generally considered safe for individuals with Polycystic Ovary Syndrome (PCOS). It works primarily by preventing ovulation and does not affect existing pregnancies. However, those with PCOS should consult their healthcare provider to discuss any specific concerns or potential interactions with other medications they may be taking.

This information is for educational purposes only and not a substitute for professional medical advice.

The evaluater verrified hallucitions or retrieval metrics for each of the cases:



