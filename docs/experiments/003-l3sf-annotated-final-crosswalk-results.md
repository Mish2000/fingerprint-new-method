# ניסוי 003 — תוצאות בירור הקשר בין `annotated_512` ל־`final_320`

**סטטוס:** הושלם ב־2026-08-29 ונעצר בשער 1 לפי הפרוטוקול הקפוא.

**הכרעה:** `NO_RELIABLE_CROSSWALK` — תמיכה ב־`H0` במובן התפעולי של הניסוי.

**משמעות מדויקת:** לא נמצאו ראיות מספיקות להקפאת crosswalk ברמת identity. זו אינה הוכחה שאין
קשר חבוי בין שני הענפים; היא קביעה שה־metadata, שני ערוצי ההתאמה והביקורת הקפואה אינם
מצדיקים שימוש בקשר כזה. לא בוצעו registration של pores ולא נוצר pseudo-ground-truth.

הפרוטוקול הוקפא ב־commit `2eef8ae` לפני צפייה בפיקסלים, והביקורת הסמנטית ומדגם 50 היחידות
הוקפאו ב־commit `989c531` לפני matching. העבודה בוצעה בענף
`experiment/003-l3sf-crosswalk`. עץ הנתונים החיצוני נקרא בלבד דרך
`fingerprint_new_method.paths`.

## 1. תשובות לשלוש שאלות הניסוי

| שאלה | תשובה | דרגת הראיה |
|---|---|---|
| א. מה תפקיד `annotated_512`? | זהו ייצוג נושא־נקבוביות, המצומד לקואורדינטות pore, בשלב שאחרי הוספת נקבוביות או באותו שלב. התאמתו ל־L3 Master Fingerprint שאחרי pores/scratches ולפני acquisition היא plausible וחזקה מבחינה סמנטית, אך אינה מוכחת ברמת הקובץ. | `supported`, לא `proved` |
| ב. האם קיים crosswalk אמין לזהויות `final_320`? | לא. 0/740 מיפויים עברו את הגדרת `STRONG`; שני הערוצים בחרו אותו top-1 רק ב־4/740. | שער 1 נכשל |
| ג. האם ניתן להעביר coordinates? | לא נבדק. תנאי העצירה אוסר לעבור ל־registration כאשר שער 1 נכשל. | `NOT_RUN` |

הסיכום המכני המלא נמצא ב־[summary.json](../../artifacts/experiment-003/summary.json).

## 2. ביקורת סמנטית

נמצאו 8,881 קבצים: 7,400 תמונות `final_320`, ‏740 תמונות `annotated_512`, ‏740 קובצי TSV
וקובץ תיאור טקסטואלי אחד. לא נמצאו archive, manifest, generation log, seed, checkpoint,
קובץ mapping או קובץ metadata נוסף.

המבנה בשני הצדדים שלם:

- בכל run יש 148 תמונות `annotated_512`;
- בכל run יש 148 זהויות מספריות של `final_320`;
- לכל זהות סופית יש בדיוק עשרת הצירופים של שתי capture groups וחמישה instances;
- אין כשל parsing ואין identity חלקית.

ב־740 קובצי JPEG נמצאו רק שדות JFIF רגילים וללא EXIF. ב־7,400 קובצי PNG לא נמצאו text
chunks או שדות פנימיים. לא נמצאו קבצים byte-identical בין הענפים. timestamps שונים בין
הענפים ומשותפים לקבוצות גדולות של קבצים ולכן הם provenance חלש בלבד, לא correspondence.

הטקסט המקומי מתאר את הסדר הבא: יצירת fingerprint בסיסי, הוספת pores/scratches וקבלת
`L3 Master fingerprint`, לאחר מכן acquisition simulation עם crop/rotation לקבלת seed, ולבסוף
image translation. הוא גם מזכיר 740 pore annotations. עם זאת, הטקסט אינו מצביע בשם או בנתיב
על קובצי ה־512 ואינו מספק crosswalk. לכן:

- **מוכח:** `annotated_512` מצומד אחד־לאחד לקבצי pore coordinates ושני הענפים מכילים אותה
  cardinality בכל run.
- **נתמך:** `annotated_512` הוא ייצוג אחרי/בזמן הוספת pores.
- **plausible אך לא מוכח:** הוא ה־L3 Master Fingerprint שלפני acquisition simulation.
- **לא נמצא:** קשר file-level או identity-level מפורש ל־`final_320`.

הפרטים וה־hash של קובץ התיאור נמצאים ב־[semantic_audit.json](../../artifacts/experiment-003/semantic_audit.json).
מספרים משותפים בשמות לא שימשו ראיה, score או tie-break מדעי.

## 3. חישוב ה־crosswalk

לכל אחת מ־740 יחידות `annotated_512` חושבו scores מול כל 148 הזהויות של אותו run בלבד,
תוך שימוש בכל עשרת ה־impressions. בסך הכול חושבו בכל ערוץ:

- 109,520 השוואות ברמת קבוצת identity;
- 1,095,200 השוואות ברמת impression;
- top-1, top-2, robust separation, support ויציבות בהשמטות.

לא נקרא אף TSV בשלב זה. לא נעשה שימוש בקואורדינטות, בצפיפות pores או בגלאי pore.

### ערוץ K — RootSIFT ו־affine-RANSAC

הערוץ השתמש ב־RootSIFT keypoints, חיפוש descriptor ברמת run, ואימות similarity-affine עם
RANSAC. מתוך 740 יחידות:

| תנאי K | עברו |
|---|---:|
| support במספר impressions | 322 |
| margin קפוא | 494 |
| leave-one-out של הקבוצה המנצחת | 421 |
| כל שלושת התנאים יחד | 311 |

ה־median של robust separation היה 9.65. כלומר K לבדו הפיק מועמדים מובחנים בחלק מן המקרים,
אך הוא אינו מספיק לפי הגדרת הניסוי ואינו crosswalk עצמאי.

### ערוץ O — ridge-orientation patches ו־affine-RANSAC

הערוץ השתמש בשדות כיוון רכסים צפופים, descriptors מקומיים נפרדים ואימות affine. תוצאותיו:

| תנאי O | עברו |
|---|---:|
| support במספר impressions | 1 |
| margin קפוא | 34 |
| leave-one-out של הקבוצה המנצחת | 31 |
| כל שלושת התנאים יחד | 0 |

ה־median של robust separation היה 0.553. תוצאה זו מראה שהערוץ לא היה discriminative מספיק
למשימה בין 148 זהויות דומות. אין לפרש את הכשל שלו כראיה שהענפים בלתי־קשורים; הוא חלק מן
הסיבה שאין ראיות מספיקות להכרעה חיובית.

### הסכמה ו־one-to-one

שני הערוצים הסכימו על top-1 רק ב־4 מתוך 740 מקרים (0.54%): אחד ב־R1, שניים ב־R2, אפס
ב־R3, אפס ב־R4 ואחד ב־R5. ה־Hungarian diagnostic הסכים עם local winner משותף בשלושה
מקרים בלבד. הוא לא שימש להפיכת מקרה עמום לחזק. בהתאם להגדרה הקפואה:

| סיווג | מספר | שיעור |
|---|---:|---:|
| `STRONG` | **0** | **0%** |
| `AMBIGUOUS` | **740** | **100%** |

טבלת כל המועמדים, החלופות והמדדים נמצאת ב־[crosswalk.csv](../../artifacts/experiment-003/crosswalk.csv),
והסיכום לפי run ו־pattern ב־[crosswalk_summary.json](../../artifacts/experiment-003/crosswalk_summary.json).

## 4. ביקורת חזותית קפואה

לפני חישוב המיפוי נבחרו באופן דטרמיניסטי 50 יחידות — עשר מכל run, עם המכסה הקפואה לפי
pattern — באמצעות הזרע `l3sf-exp003-crosswalk-review-v1`. לא בוצעה החלפה לפי איכות.

מאחר שבמדגם לא היה אף מקרה של consensus בין הערוצים, לא הייתה “identity נבחרת” במובן של
`STRONG`. כדי להשלים את הביקורת בלי לשנות mapping, הוצג באופן עיוור ה־Hungarian diagnostic
שהוגדר מראש מול המועמד המקומי הראשון השונה ממנו. לכל צד הוצגו impressions במיקומים הקנוניים
1, 6 ו־10; סדר A/B נקבע ב־hash; identifiers הוסתרו. contingency זה הוא diagnostic בלבד ואינו
יכול לשדרג מיפוי עמום.

| סיווג ביחס למועמד הדיאגנוסטי | מספר | שיעור |
|---|---:|---:|
| `CLEAR` | 28 | 56% |
| `AMBIGUOUS` | 12 | 24% |
| `NOT_MATCH` | 10 | 20% |
| `CLEAR` או `AMBIGUOUS` | 40 | 80% |

שני שערי הביקורת נכשלו: נדרשו לפחות 38/50 `CLEAR` ולפחות 45/50 `CLEAR` או `AMBIGUOUS`.
כל ההכרעות והנימוקים נשמרו ב־[crosswalk_review.csv](../../artifacts/experiment-003/crosswalk_review.csv).
הלוחות המכילים pixels נשארו מקומיים תחת `evidence-pixels` המוחרג מ־Git.

## 5. שער 1 ותנאי העצירה

שער 1 נכשל בכל שלושת מוקדי הראיה:

1. נדרשו לפחות 703 correspondences חזקים; התקבלו 0.
2. שני הערוצים כמעט שאינם מסכימים, והערוץ המבוסס ridge orientation אינו עובר את תנאיו
   כערוץ עצמאי.
3. הביקורת הקפואה אינה עוברת אף אחד משני הספים החזותיים.

לכן ה־crosswalk לא הוקפא כמיפוי שמותר להשתמש בו. `crosswalk.csv` הוא טבלת תוצאות עמומות,
לא label table.

## 6. geometry ו־pore transfer

לא נבחרו 30 identities, לא נבחרו 60 זוגות, לא חושב transform ולא נקרא TSV לצורך transfer.
הקבצים [registration_summary.csv](../../artifacts/experiment-003/registration_summary.csv) ו־
[pore_transfer_review.csv](../../artifacts/experiment-003/pore_transfer_review.csv) מכילים schema
בלבד ומוכיחים במפורש שהשלב לא רץ. שער 2 מסומן `NOT_RUN`.

אין ליצור על סמך ניסוי זה pseudo-ground-truth של pores עבור `final_320`.

## 7. מגבלות הפרשנות

- תוצאת `H0` היא הכרעת evidence, לא טענה ששני הענפים נוצרו בוודאות באופן בלתי־תלוי.
- ערוץ O לא הראה discrimination מספק; ייתכן שמימוש מבני אחר היה מפיק evidence שונה, אך
  שינוי השיטה לאחר חשיפת התוצאות היה מפר את ההקפאה ואינו חלק מניסוי 003.
- ה־visual review השווה שתי קבוצות מועמדות בלבד ואינו חיפוש ידני בין 148 identities.
- לא בוצע חיפוש בין runs, בהתאם למודל המועמדים שנקבע מראש.
- cardinality, סדר מספרי ויכולת לפתור assignment אינם ראיות correspondence.

## 8. תוצרים ושחזור

מטריצות ה־score הגדולות נשמרו מקומית בחמישה קובצי NPZ תחת `local-large`. לכל קובץ shape
של 148×1,480 לכל metric ושמונה metrics בכל אחד משני הערוצים. שמות, sizes ו־SHA-256 נמצאים
ב־[score_matrices.manifest.json](../../artifacts/experiment-003/score_matrices.manifest.json).

שחזור:

```powershell
& .\.conda-env\python.exe .\scripts\experiment_003_semantic_audit_and_freeze.py
& .\.conda-env\python.exe .\scripts\experiment_003_build_crosswalk.py --overwrite
& .\.conda-env\python.exe .\scripts\experiment_003_prepare_crosswalk_review.py
& .\.conda-env\python.exe .\scripts\experiment_003_finalize_crosswalk_review.py
& .\.conda-env\python.exe .\scripts\experiment_003_finalize.py
```

הניסוי נעצר כאן. לא אומן detector, descriptor, matcher, classifier או synthetic detector,
ולא בוצע מבחן 6,000 ההשוואות.
