"""Why each of the 32 inputs is or is not used by the model, in plain words.

Facts are from the 519 training students. "Rank" = position of the input's information score among all 45 columns
(1 = tells us the most about pass / at risk). The model keeps the top 5, because in cross-validation keeping 5 columns
did better than keeping 10, 20 or all 45 (see results/results.json, experiments -> features).
Shown in the app as the (?) tooltip on every input and in the "Feature choices" tab.
"""

REASONS = {
    # ---- the 5 inputs the model uses ----
    "G2": "Rank 1, the strongest signal. The 2nd-period grade is the closest thing to the final grade (correlation 0.92).",
    "G1": "Rank 2. Students with a 1st-period grade of 8 or less fail 81% of the time; with 11 or more, almost never.",
    "failures": "Rank 3. No past failures: 9% fail. One past failure: 45%. Two or more: about 60%.",
    "school": "Rank 4. 31% of students at Mousinho da Silveira (MS) fail, against 7% at Gabriel Pereira (GP).",
    "higher": "Rank 5. Students who don't want higher education fail 46% of the time, against 12% of those who do.",
    # ---- not used ----
    "absences": "Rank 6. Students with more than 10 absences fail more (24% vs 14%), but keeping more than the top 5 "
                "columns did not improve the model, and many absences usually already show up as low grades.",
    "studytime": "Rank 7. Studying under 2 hours a week: 26% fail; over 10 hours: 4%. But students who study little "
                 "usually already have low G1/G2, which the model uses.",
    "Fedu": "Rank 8. Father with only primary education: 27% fail vs 10% with higher education. A weaker signal than the grades.",
    "Medu": "Rank 9. Mother with only primary education: 27% fail vs 8% with higher education. A weaker signal than the grades.",
    "famrel": "Rank 10. Very bad family relations: 42% fail, but that is only 19 students. Too weak and too rare to help.",
    "goout": "Rank 11. Going out very little or very often both fail a bit more (24-26% vs 11%). A weak signal.",
    "age": "Rank 12. Older students fail more mostly because they repeated a year, and 'past failures' already captures that.",
    "freetime": "Rank 13. Only students with very high free time stand out (29% fail). A weak signal once grades are known.",
    "Walc": "Rank 14. Heavy weekend drinkers fail more (25% vs 12%), but the grades already show it.",
    "reason": "Rank 15. Small differences between reasons, and they mostly reflect which school the student chose.",
    "Mjob": "Rank 16. Mothers who are teachers: 5% fail. This overlaps with mother's education.",
    "Dalc": "Rank 17. Heavy workday drinking is rare (15 students) and overlaps with weekend drinking.",
    "address": "Rank 19. Rural students fail more (21% vs 13%), but they also have lower grades and most of them go to "
               "MS. Among students with the same G2 grade, rural and urban students fail at about the same rate, so "
               "address adds nothing new once grades and school are known.",
    "traveltime": "Rank 22. Long travel goes with rural address and school, which are covered by other inputs.",
    "Fjob": "Rank 23. Fathers who are teachers: 4% fail, but only 26 students. Overlaps with father's education.",
    "guardian": "Rank 24. Small differences (father 10%, mother 17%, other 21%) in small groups.",
    "romantic": "Rank 25. In a relationship: 19% fail vs 13%. A weak signal that disappears once grades are known.",
    "internet": "Rank 26. No internet: 21% fail vs 14%, but no internet is much more common in rural homes, so it "
                "mostly repeats address and school.",
    "activities": "Rank 27. Extra-curricular activities: 13% fail vs 18%. Weak.",
    "sex": "Rank 28. Boys fail slightly more (19% vs 13%). Weak, and leaving it out is also fairer.",
    "health": "Rank 29. No clear pattern (very bad health 19%, good 12%, very good 17%).",
    "famsup": "Rank 33. Family support: 14% vs 17%. Almost no difference.",
    "famsize": "Rank 34. Small families 13% vs large 16%. Almost no difference.",
    "schoolsup": "Rank 36. 13% vs 16%. Extra support is given to weaker students, so it is not a clear signal.",
    "nursery": "Rank 38. Went to nursery school 16% vs not 14%. No real difference.",
    "Pstatus": "Rank 39. Parents apart 17% vs together 15%. No real difference.",
    "paid": "Rank 41, the lowest. Paid extra classes 14% vs 16%, and only 28 students took them.",
}
