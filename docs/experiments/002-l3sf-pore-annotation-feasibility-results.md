# ניסוי 002 — בדיקת כשירות L3-SF לפיתוח ייצוג נקבוביות Level-3

**סטטוס:** הושלם ב־2026-08-28. הניסוי נעצר לאחר הכרעה; לא פותח detector, descriptor או matcher.

**תחום ההכרעה:** תת־הקבוצה המקומית של 740 תמונות ה־master המסומנות, כבסיס ראשוני למשימת pore localization בלבד. מערך המקור המוגדר באמצעות `FINGERPRINT_DATASETS_ROOT` נקרא בלבד; כל התוצרים נכתבו תחת תיקיית העבודה של הפרויקט.

## 1. מבנה מערך הנתונים

בפועל נמצאו תחת `L3SF_V2` שלושה פריטים עליונים: `L3-SF`, התיאור המקומי `L3-SF - Level Three Synthetic Fingerprint Generation.txt`, ו־`Pore ground truth`.

| ענף | תמונות | פורמט וממדים | מבנה שנצפה |
|---|---:|---|---|
| `L3-SF/R1..R5` | 7,400 | PNG, ‏320×240×3 | 1,480 תמונות בכל run; שם `numeric_identity_capture_group_instance.png` |
| `Pore ground truth/Fingerprint Images/R1..R5` | 740 | JPG, ‏512×512×3 | 148 תמונות בכל run; תמונת master מסומנת אחת לכל מזהה קנוני מקומי |
| **סה״כ תמונות** | **8,140** | 7,400 PNG ו־740 JPG | כל התמונות פוענחו בהצלחה |

בענף המלא, מזהה identity בטוח הוא `R{run}/{numeric_identity}`. קיימות 740 identities, ולכל אחת בדיוק עשר תמונות: שני ערכי `capture_group` וחמישה `instance` בכל קבוצה. כל 740 הזהויות כוללות את כל צירופי 2×5; אין identity חלקית.

בענף המסומן קיימות 740 תמונות: 370 `right_loop`, ‏300 `whorl`, ‏25 `left_loop`, ‏30 `plain_arch` ו־15 `tented_arch`. השם הקנוני הוא `R{run}/{filename_stem}`; לדוגמה `R2/38_right_loop`. אין על הדיסק קובץ crosswalk מפורש בין שמות אלה לבין הזהויות המספריות בענף המלא, ולכן אין להסיק correspondence מן המספרים בשם בלבד.

ה־inventory המלא, לרבות נתיבי קבצים, SHA-256, dimensions ושדות identity, נמצא ב־[dataset_images.csv](../../artifacts/experiment-002/inventory/dataset_images.csv) וב־[dataset_inventory.json](../../artifacts/experiment-002/inventory/dataset_inventory.json).

## 2. מבנה ה־annotations

לכל אחת מ־740 תמונות ה־JPG קיים בדיוק קובץ TSV אחד באותו `R` ובאותו stem. שורת הכותרת היא `x<TAB>y`, וכל רשומה מכילה שני מספרים שלמים בלבד. נמצאו 306,009 רשומות, 282–551 לתמונה; החציון הוא 410 והממוצע 413.526.

הייצוג הוא **נקודה יחידה** של מיקום/מרכז נקבובית, לא contour, מסכה או bounding region. אין שדה confidence, אין סוגי annotation נוספים, ואין label המבחין בין open ל־closed pore. אף שהתיאור המקומי של המערך מזכיר חזותית open ו־closed pores, הבחנה זו אינה קיימת בקובצי האמת.

מערכת הקואורדינטות שאושרה היא:

- origin בפינה השמאלית־עליונה;
- `x` הוא column ו־`y` הוא row;
- יש להשתמש בערכים שב־TSV ישירות ככתובות פיקסל, ללא הפחתת 1;
- הטווח שנצפה בפועל בשני הצירים הוא 1–511 בתמונה של 512×512.

הטווח לבדו אינו מכריע אם המקור 0- או 1-based. לכן נבדקה תגובת הבהירות המצטברת בכל 306,009 הנקודות תחת כל היסט שלם קבוע בטווח ±4 פיקסלים. נקבוביות נראות כפתחים בהירים על ridge כהה, והתגובה המרבית התקבלה בדיוק ב־`dx=0, dy=0`: ממוצע grayscale ‏142.865 לעומת 136.609 בהפחתת פיקסל משני הצירים ו־121.703 בהחלפת הצירים. גם overlays בפיקסלי המקור התאימו ישירות. הבדיקה מתועדת ב־[coordinate_convention_check.json](../../artifacts/experiment-002/inventory/coordinate_convention_check.json); היא בדיקת registration תיאורית, לא detector.

ה־inventory ברמת קובץ ורשומה נמצא ב־[annotation_files.csv](../../artifacts/experiment-002/inventory/annotation_files.csv) וב־[annotations.csv](../../artifacts/experiment-002/inventory/annotations.csv).

## 3. שלמות ותקינות

בדיקת כל הקבצים נתנה את התוצאות הבאות:

- 0 כשלים בפענוח תמונה, 0 קובצי TSV פגומים ו־0 שמות שאינם ניתנים לפענוח לפי המבנה שנצפה;
- 0 annotation files מיותמים ו־0 תמונות מסומנות ללא TSV תואם;
- 0 מתוך 306,009 קואורדינטות מחוץ לתחום;
- 0 קבוצות של תמונות byte-identical ו־0 קבוצות של קובצי TSV byte-identical לפי SHA-256;
- 0 התנגשויות במזהי sample קנוניים ו־0 identities חלקיות בענף 7,400 התמונות.

נמצאו **241 רשומות קואורדינטה כפולות במדויק**, כלומר רשומות עודפות מעבר למיקום הייחודי הראשון בתוך אותו TSV. הן מהוות 0.0788% מכל הרשומות, מופיעות ב־210 מתוך 740 תמונות, והמקסימום הוא שלוש רשומות עודפות בתמונה. אין זו בעיית corruption מערכתית, אך לפני אימון או scoring חובה לבצע deduplication לפי `(canonical_image_id, x, y)` כדי שנקודה אחת לא תיספר פעמיים. במדגם הביקורת הקפוא לא נבחר אותו מיקום כפול פעמיים.

הממצא המבני היחיד שאינו שגיאת קובץ הוא היעדר crosswalk בין annotated masters לענף הזהויות המספרי. מגבלה זו מטופלת בפרוטוקול הפיצול בסעיף 7. כל ממצאי התקינות נמצאים ב־[integrity_findings.json](../../artifacts/experiment-002/inventory/integrity_findings.json).

## 4. בחירת מדגם הביקורת

המדגם הוקפא לפני צפייה בפיקסלים שלו, עם הזרע `l3sf-exp002-blind-audit-v1`. בכל אחד מ־R1..R5 דורגו התמונות באמצעות `SHA256(seed|image|R/stem)` ונבחרו ארבע `right_loop`, שלוש `whorl`, אחת `left_loop`, אחת `plain_arch` ואחת `tented_arch`. כך התקבלו עשר תמונות לכל run וייצוג מכוון גם ל־patterns הנדירים. בתוך כל תמונה דורגו הרשומות באמצעות SHA-256 של הזרע, המזהה, מספר השורה והקואורדינטה, ונבחרו 20 הראשונות. לא נעשתה החלפה לפי איכות.

עשר תמונות שהוצגו קודם רק כדי לפרש את מבנה הדיסק והזהויות הוחרגו מראש מן ההגרלה: `R1/18_right_loop`, ‏`R1/1_left_loop`, ‏`R1/1_plain_arch`, ‏`R1/1_right_loop`, ‏`R1/1_whorl`, ‏`R1/44_right_loop`, ו־`R2..R5/1_whorl`. ההחרגה שומרת על עיוורון ואינה מבוססת איכות.

רשימת 50 התמונות שנבחרו:

| מכסה | R1 | R2 | R3 | R4 | R5 |
|---:|---|---|---|---|---|
| 1 | S01 `2_left_loop` | S11 `1_left_loop` | S21 `3_left_loop` | S31 `5_left_loop` | S41 `4_left_loop` |
| 2 | S02 `2_plain_arch` | S12 `3_plain_arch` | S22 `3_plain_arch` | S32 `4_plain_arch` | S42 `1_plain_arch` |
| 3 | S03 `12_right_loop` | S13 `25_right_loop` | S23 `38_right_loop` | S33 `21_right_loop` | S43 `28_right_loop` |
| 4 | S04 `73_right_loop` | S14 `50_right_loop` | S24 `57_right_loop` | S34 `33_right_loop` | S44 `40_right_loop` |
| 5 | S05 `9_right_loop` | S15 `38_right_loop` | S25 `2_right_loop` | S35 `14_right_loop` | S45 `16_right_loop` |
| 6 | S06 `4_right_loop` | S16 `54_right_loop` | S26 `5_right_loop` | S36 `44_right_loop` | S46 `64_right_loop` |
| 7 | S07 `2_tented_arch` | S17 `2_tented_arch` | S27 `3_tented_arch` | S37 `3_tented_arch` | S47 `1_tented_arch` |
| 8 | S08 `42_whorl` | S18 `56_whorl` | S28 `19_whorl` | S38 `59_whorl` | S48 `28_whorl` |
| 9 | S09 `21_whorl` | S19 `21_whorl` | S29 `16_whorl` | S39 `9_whorl` | S49 `21_whorl` |
| 10 | S10 `48_whorl` | S20 `43_whorl` | S30 `21_whorl` | S40 `54_whorl` | S50 `33_whorl` |

הנתיבים, גיבובי SHA-256 המלאים של התמונה וה־TSV, דירוגי הבחירה וכל 1,000 הרשומות הקפואות נמצאים ב־[review_sample_manifest.json](../../artifacts/experiment-002/review/review_sample_manifest.json), ב־[review_sample_images.csv](../../artifacts/experiment-002/review/review_sample_images.csv) וב־[review_annotation_manifest.csv](../../artifacts/experiment-002/review/review_annotation_manifest.csv).

## 5. תוצאות הביקורת החזותית

כל 1,000 הנקודות נבדקו בחיתוכי 48×48 מפיקסלי המקור, שהוגדלו ב־nearest-neighbor לצפייה. טבעת אדומה שימשה display בלבד ולא כיסתה את פיקסל המרכז. לכל תמונה נבדקו גם overview וגם 20 החיתוכים. מדדי ניגודיות קבועים חושבו רק כדי לנווט לבדיקה חוזרת של חריגים לאחר הקפאת המדגם; הם לא קבעו label ולא שימשו classifier.

| סיווג | R1 | R2 | R3 | R4 | R5 | סה״כ | שיעור |
|---|---:|---:|---:|---:|---:|---:|---:|
| ברור (`CLEAR`) | 190 | 185 | 187 | 189 | 185 | **936** | **93.6%** |
| עמום (`AMBIGUOUS`) | 10 | 13 | 13 | 11 | 13 | **60** | **6.0%** |
| לא תואם (`NOT_MATCH`) | 0 | 2 | 0 | 0 | 2 | **4** | **0.4%** |
| ברור או עמום | 200 | 198 | 200 | 200 | 198 | **996** | **99.6%** |

מתוך 60 המקרים העמומים, 54 היו קרובים עד שמונה פיקסלים משפת התמונה ולכן ההקשר הדרוש לאימות מדויק נחתך; בשישה נוספים נראה מבנה מתאים אך לא ניתן היה להפריד בביטחון נקבובית יחידה במרכז. ארבעת המקרים הלא־תואמים היו:

- S15, ‏`R2/38_right_loop`: שורות TSV ‏50 ב־(56,452) ו־43 ב־(48,449);
- S49, ‏`R5/21_whorl`: שורה 25 ב־(465,409);
- S50, ‏`R5/33_whorl`: שורה 270 ב־(47,417).

בארבעתם לא נראה בכתובת מבנה ridge-pore מתאים; האזור היה רקע/מרקם חלש ללא נקבובית נפרדת. שיעור `CLEAR_OR_AMBIGUOUS` עובר את שער 90%, ושיעור `CLEAR` עובר את שער 75%, בפער גדול וללא שינוי ספים.

כל 1,000 ההכרעות, הסיבה, נתיב הראיה ומדדי הניווט נשמרו ב־[review_ratings.csv](../../artifacts/experiment-002/review/review_ratings.csv); הסיכום לפי run, pattern ותמונה נמצא ב־[review_summary.json](../../artifacts/experiment-002/review/review_summary.json).

## 6. מאפיינים מרחביים

הבדיקה בוצעה על כל 306,009 הרשומות, לא רק על המדגם:

- annotations לתמונה: מינימום 282, ‏p05=333, חציון 410, ‏p95=504, מקסימום 551;
- צפיפות ל־100,000 פיקסלי fingerprint משוערים: מינימום 108.19, חציון 157.54, ממוצע 158.51 ומקסימום 210.19;
- nearest-neighbor: ‏p05=3.162 px, חציון 15.620 px, ‏p95=27.203 px וממוצע 15.418 px. ערכי 0 נובעים מן הכפילויות המדויקות;
- עשרת bins של X מכילים 29,792–31,799 נקודות, ושל Y ‏29,976–31,150 — ללא שיא אזורי חריג או תבנית grid;
- 38,263 נקודות (12.504%) נמצאות עד 16 px משפה, ו־57,756 (18.874%) עד 5% מממד התמונה. שיעורים אלה קרובים מאוד לציפייה הגאומטרית של פיזור אחיד בשטח ריבועי: 12.109% ו־19.0% בהתאמה, ולכן אינם מצביעים על clipping עודף.

מסכת fingerprint לצורך הצפיפות היא אומדן גס: Otsu לרכסים כהים, morphological close בגודל 31×31, dilation בגודל 9×9 ושמירת רכיבים גדולים. כיוון שה־master ממלא כמעט את כל 512×512, שטח המסכה הממוצע הוא כ־99.5% מן התמונה. לכן הצפיפות היא בדיקת artifact השוואתית ולא אומדן אנטומי מוחלט.

לא נמצאו displacement קבוע, החלפת צירים, concentration חריג או grid מלאכותי. הממצא המרחבי היחיד הדורש טיפול הוא deduplication של 241 הרשומות הזהות. הנתונים המלאים נמצאים ב־[spatial_summary.json](../../artifacts/experiment-002/inventory/spatial_summary.json) וב־[spatial_per_image.csv](../../artifacts/experiment-002/inventory/spatial_per_image.csv).

## 7. סכנת leakage

בענף המלא יש עשר דוגמאות לכל `R/numeric_identity`; לכן split אקראי ברמת image יהיה שגוי. אם ענף זה ישמש בעתיד, יחידת הפיצול חייבת להיות כל identity, וכל עשר התמונות שלה חייבות להישאר באותה partition.

בתת־הקבוצה המסומנת קיימת תמונת master אחת לכל מזהה קנוני `R/stem`. עבור משימת localization ניתן לבצע split מדעי ברמת master/identity: כל תמונת master, כל החיתוכים וכל derivative שיופק ממנה נשארים יחד; שום master אינו מופיע בשתי partitions. קיימות 740 יחידות כאלה, ולכן אפשר לבצע stratification לפי `R` ו־pattern בלי לפצל identity. בהתאם להוראות, לא נוצר בניסוי זה split סופי.

היעדר crosswalk בין annotated master לבין הענף המספרי מחייב כלל בטיחות מפורש: **אין לערב את שני הענפים באותו train/validation/test protocol עד שייווצר או יאומת crosswalk אמין.** תחת מגבלה זו, annotated-only master-disjoint split אפשרי ואינו דורש filtering לפי איכות. בכך מתקיים שער ה־identity-disjoint עבור תחום ההכרעה של ניסוי זה.

## 8. כשירות למדידה

ניתן להגדיר target חד־משמעי: detector מחזיר אוסף נקודות `(x,y)` בתמונה 512×512, באותה מערכת ישירה של ה־TSV. לפני scoring מסירים כפילויות זהות מן ה־ground truth.

הצעת המדידה הראשונית, שאינה מכוילת על ביצועי detector כלשהו:

1. prediction ו־ground truth מותאמים אם המרחק האוקלידי ביניהם אינו עולה על **4 px** ברזולוציית 512×512 המקורית. זהו 0.0078125 מממד ציר אחד.
2. ההתאמה היא one-to-one. תחילה ממקסמים cardinality של ההתאמות בתוך ה־tolerance, ובין פתרונות בעלי אותו cardinality ממזערים את סכום המרחקים.
3. localization error מחושב רק על זוגות שהותאמו; prediction שלא הותאם הוא false positive ו־GT שלא הותאם הוא false negative.
4. כל resize או crop עתידי חייב לשמור transform דטרמיניסטי חזרה למערכת 512×512 לפני המדידה.

ה־4 px מוצע משום שהוא קטן משמעותית ממרחק ה־nearest-neighbor החציוני 15.62 px, אך מאפשר אי־דיוק תיוג/חיזוי קטן סביב מרכז הפתח. הוא לא נבחר כדי לשפר metric — אין בניסוי detector ותוצאות detector. לפני ניסוי אלגוריתמי יש לרשום אותו מראש; אפשר לדווח גם sensitivity תיאורי ב־2 וב־6 px, אך אסור לבחור tolerance לפי test performance.

לא נמצא metadata של PPI או scale פיזיקלי אמין עבור תמונות ה־master, ולכן המדידה נשארת pixel-based. ה־annotations מאפשרים לחשב בעתיד Precision, Recall, F1, mean localization error בפיקסלים, false positives per image ו־recall per image. אין להשתמש ב־pixel accuracy בינארי, משום שרוב הפיקסלים אינם נקבוביות.

## 9. ממצאים בלתי צפויים

1. אין crosswalk מפורש בין 740 ה־annotated masters לבין 740 הזהויות המספריות בעלות עשר הדוגמאות. המספרים בשמות אינם בסיס בטוח למיפוי.
2. נמצאו 241 רשומות `(x,y)` כפולות מדויקות, מפוזרות על פני 210 תמונות. השיעור זעיר אך ללא deduplication הוא עלול לעוות loss או scoring נקודתי.
3. ערכי הקואורדינטות מתחילים ב־1 אך שימוש ישיר, לא הפחתת 1, נתן registration מצטבר מיטבי. בדיקה זו מנעה הנחה שגויה על coordinate origin.
4. הפיזור בצירים ובקרבת השוליים כמעט אחיד גאומטרית; לא נמצאה concentration מלאכותית אף שהנקבוביות סינתטיות.
5. ארבעה annotations לא־תואמים אכן קיימים במדגם, אך 996/1,000 נותרו ברורים או עמומים, והחריגות אינן מצביעות על כשל correspondence מערכתי.
6. התיאור המקומי מזכיר open/closed pores, אך קובצי ה־TSV אינם מקודדים את ההבחנה ולכן לא ניתן למדוד אותה.

## 10. הכרעה

`STRONG_PASS`

| תנאי השער | תוצאה |
|---|---|
| מיפוי image↔annotation חד־משמעי | כן — 740/740 זוגות exact stem בתוך אותו `R` |
| אין corruption/integrity מערכתי | כן — 0 decode, parse, missing או out-of-bounds failures |
| לפחות 90% ברור או עמום | כן — 99.6% |
| לפחות 75% ברור | כן — 93.6% |
| identity-disjoint split אפשרי | כן — annotated-only split ברמת master; derivatives נשארים יחד |
| localization metric חד־משמעי | כן — נקודות direct `(x,y)`, one-to-one ו־tolerance קבוע |
| אין artifact מבני הפוסל את המשימה | כן — הכפילויות זעירות וניתנות ל־deduplication; לא נמצא grid/displacement |

ההכרעה חלה על L3-SF כ־ground-truth development dataset **ראשוני וסינתטי** ל־pore localization, תחת פרוטוקול annotated-only והגבלות הדוח. הסיכום המכונתי נמצא ב־[summary.json](../../artifacts/experiment-002/summary.json).

## 11. פרשנות

התוצאה מאפשרת לשכבת התכנון להגדיר ניסוי המשך נפרד לפיתוח גלאי נקבוביות על 740 ה־masters המסומנים. ניסוי כזה צריך להקפיא מראש master-disjoint train/validation/test split, לבצע deduplication, לשמור את מערכת 512×512, לקבוע 4 px כ־tolerance ראשי ולהשתמש במדדי detection/localization המוגדרים בסעיף 8.

ה־annotations מספקים target מספיק ברור ואיכות חזותית גבוהה לפיתוח ראשוני; הם אינם פותרים את שאלת generalization לטביעות אמיתיות. ה־crosswalk לענף 7,400 התמונות צריך להישאר מחוץ לניסוי הבא או להיבדק במשימה ייעודית לפני שימוש. בהתאם לתנאי העצירה, לא נבנה כאן baseline מכויל, detector, descriptor או matcher.

## 12. מה לא נטען

- L3-SF הוא מערך synthetic; התוצאה אינה מוכיחה שהתפלגות, appearance או semantics של הנקבוביות מייצגים אנטומיה אנושית אמיתית.
- הצלחה עתידית על annotations של L3-SF אינה מוכיחה generalization ל־SD300.
- הניסוי אינו מוכיח שיפור recognition, כוח הבחנה ביומטרי או ירידה בשיעורי שגיאה.
- הניסוי אינו מוכיח יכולת real-vs-synthetic detection.
- הניסוי אינו מעריך matcher, descriptor, fusion עם minutiae או השוואה מול אלגוריתמים קיימים.
- בדיקת ההיסט המצטברת ומדדי הניגודיות אינם detector ואינם baseline ביצועים.
- לא אומן model, לא כויל threshold על SD300, לא הורץ מבחן 6,000 ההשוואות ולא שונה `fingerprint-benchmark`.
- ההכרעה אינה מתירה לערב את הענף המספרי והענף המסומן בלי crosswalk, ואינה מבטלת את חובת ה־deduplication וה־identity-disjoint split.

תוצרי הליבה המכונתיים נמצאים תחת [artifacts/experiment-002](../../artifacts/experiment-002/): inventory, integrity, spatial analysis, המדגם הקפוא, 1,000 ratings והסיכום הסופי.
