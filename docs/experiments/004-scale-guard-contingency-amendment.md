# ניסוי 004 — תוספת קפואה: contingency לגבול ה־scale factor ב־SD300

מסמך זה נכתב ונכנס ל־Git **לפני כל גישה לקבצי SD300** בניסוי 004. הוא אינו משנה את
[הפרוטוקול הקפוא](004-pore-localization-and-sd300-transfer-protocol.md) שהוקפא ב־commit
`6b95834a7ad0ce1b40176db49a6867a95ce10f42`, ואינו משנה אף שער הכרעה.

## 1. הבעיה שזוהתה

סעיף 9 בפרוטוקול הקפוא קובע:

> לכל SD300 image scale factor הוא `target_period / estimated_period`; factor מחוץ `[0.20,1.50]`
> או אומדן לא אמין מסומן `PREPROCESSING_FAILURE`.

ה־target period הוקפא כ־`34.0 px`, כ־median של 318 אומדנים אמינים מתוך 440 תמונות L3-SF **train
בלבד** (`ridge_period_train_estimates.json`; טווח 23.5–60.0 px). כלומר `annotated_512` הוא domain
ממוזג מאוד — שקול בקירוב ל־1700 ppi.

ridge period טיפוסי של טביעה אמיתית הוא כ־0.5 מ״מ, ולכן צפוי:

| רזולוציית SD300 | ridge period צפוי | `34 / period` | בתוך `[0.20,1.50]`? |
| --- | --- | --- | --- |
| 1000 ppi (ראשית) | ~16–24 px | ~1.4–2.1 | לעיתים קרובות **לא** |
| 2000 ppi (sensitivity) | ~32–48 px | ~0.7–1.1 | כן |

הגבול הקפוא אינו סימטרי, ולכן הוא עלול לפסול חלק גדול מן **הרזולוציה הראשית** לפני שה־detector
בכלל רץ. במקרה כזה שער ב' היה מוכרע כ־`TRANSFER_INCONCLUSIVE` מסיבה טכנית של preprocessing ולא
מסיבה מדעית.

מפרט הניסוי עצמו מגדיר את תנאי הכישלון אחרת — לפי **אמינות האומדן** ("אם האומדן אינו אמין, סמן
preprocessing failure"), ולא לפי גודל ה־factor. הגבול `[0.20,1.50]` הוא הידוק שנוסף במימוש.

## 2. מצב העיוורון בעת כתיבת המסמך

- לא נקרא, לא פוענח ולא נמדד אף קובץ SD300 בניסוי 004.
- לא קיים אף artifact בשם `sd300_*` תחת `artifacts/experiment-004/`.
- לא נפתחו דירוגי Experiment 001.
- ה־test של L3-SF טרם נפתח; האימון עדיין רץ.

הזיהוי נובע מן ה־target period של L3-SF ומידע כללי על ridge frequency בלבד.

## 3. ה־contingency הקפוא

מרגע החתימה על מסמך זה, ולפני כל גישה ל־SD300, נקבע:

1. **הניתוח הראשי נשאר הפרוטוקול הקפוא.** תמונה שה־factor שלה מחוץ `[0.20,1.50]` נחשבת
   `PREPROCESSING_FAILURE` בניתוח הראשי, וזוג שמערב אותה אינו נכנס למדד הראשי ולשער ב'.
2. **בנוסף יחושב variant משני יחיד**, שבו `PREPROCESSING_FAILURE` נקבע אך ורק לפי כלל האמינות
   של הפרוטוקול (לפחות 5 tiles אמינים, `MAD/median <= 0.25`), עם sanity guard רחב
   `[0.20,3.00]` שנועד רק לתפוס אומדן אבסורדי.
3. **ה־variant אינו יכול לשנות את ההכרעה.** שער ב' וההכרעה הסופית מחושבים מן הניתוח הראשי בלבד.
   ה־variant מדווח כ־sensitivity ומסומן exploratory.
4. שאר הצינור זהה לחלוטין בשני הניתוחים: אותו target period, אותו estimator, אותו
   `INTER_AREA`/`INTER_CUBIC`, אותם weights, threshold, NMS, tiling, registration, tolerance
   ו־bootstrap seed. שום pore coordinate, density או repeatability אינו משתתף בבחירת scale.
5. מכיוון שתמונה שמתקבלת בשני הכללים עוברת בדיוק את אותו resize, ה־preprocessing, ה־heatmaps
   וה־registrations מחושבים פעם אחת, ושני הניתוחים נבדלים אך ורק בסינון הזוגות. כך הניתוח הראשי
   זהה bit-for-bit לתוצאה שהפרוטוקול הקפוא היה מפיק.
6. שתי התוצאות ידווחו במלואן, גם אם הן סותרות. אין לבחור ביניהן בדיעבד.

## 4. מה עדיין אסור

התוספת אינה מתירה training, fine-tuning, threshold calibration, checkpoint selection,
architecture selection או בחירת scale לפי SD300. היא אינה משנה את `target_period_px`, את
ה־tolerance, את ספי שער א' או את חמשת תנאי שער ב'.
