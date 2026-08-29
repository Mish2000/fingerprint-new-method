# ניסוי 004 — תוצאות pore localization והעברה קפואה ל־SD300

**סטטוס:** הושלם ב־2026-08-29 עם הכרעה **`SYNTHETIC_LOCALIZATION_ONLY`**.

הפרוטוקול הוקפא ב־commit `6b95834a7ad0ce1b40176db49a6867a95ce10f42` לפני אימון. ה־test של L3-SF לא שימש לבחירת architecture, checkpoint, threshold, NMS, loss או preprocessing.
Audit העיוורון נשמר ב־`blindness_audit.json`; לא נפתחו item-level ratings של Experiment 001 לפני freeze של SD300.

## נתונים ו־ground truth

החלוקה הקפואה כוללת 88/30/30 קבוצות leakage ו־440/150/150 תמונות train/validation/test. כל חמשת ה־runs של אותו `pattern + local_index` נשארו יחד. הוסרו בדיוק 241 רשומות coordinate כפולות; annotations אחרים, לרבות שפתיים ועמומים, נשארו במדד.

Ridge period המטרה, שנאמד מ־train בלבד, היה `34.00 px`.

## אימון ואירועי runtime

אומנו אותם 7,240,225 פרמטרים בשלושת ה־seeds ['40401', '40402', '40403']. ה־checkpoints נבחרו לפי validation loss בלבד. נשמרו 10 ניסיונות בסך הכול; 7 ניסיונות שלא הושלמו נשארו ב־manifest ואינם מוסתרים.

| seed | epochs | best epoch | best validation loss | דקות |
|---:|---:|---:|---:|---:|
| 40401 | 57 | 57 | 0.00185286 | 70.4 |
| 40402 | 42 | 41 | 0.00185513 | 49.8 |
| 40403 | 57 | 45 | 0.00185324 | 59.2 |

כל seed נעצר בדיוק 12 epochs לאחר השיפור האחרון שגדול מ־`early_stopping_min_delta`, ולכן ה־checkpoint הנבחר יושב בסוף הריצה: כלל ה־checkpoint משתמש בסף `1e-8` וכלל ה־early stopping בסף `1e-5`, כפי שנקבע בפרוטוקול לפני האימון.

סיבות הניסיונות שלא הושלמו: `CONSOLE_CLOSE_ABORT`×1, `EXTERNAL_PROCESS_TERMINATION`×1, `INTERRUPTED_CODE_AUDIT`×1, `INTERRUPTED_MEMORY_LAYOUT_OPTIMIZATION`×1, `INTERRUPTED_RESUME_VERIFICATION`×1, `INTERRUPTED_RUNTIME_OPTIMIZATION`×1, `NATIVE_RUNTIME_ABORT`×1. שניים מהם היו aborts של הסביבה (`forrtl: error (200)`) שנגרמו מסגירת חלון console; אחד היה הפסקה מכוונת שאימתה את מנגנון ה־resume; היתר היו תיקוני מימוש ו־runtime לפני כל גישה ל־test. הוספת ה־resume ברמת epoch היא שינוי תפעולי: סדר ה־batch נגזר מ־seed לכל epoch, ה־augmentation נגזרת מ־`(seed, epoch, image_id)`, ואין שכבות סטוכסטיות, ולכן הריצה שחודשה שחזרה בדיוק את ערכי ה־validation שלפני ההפסקה (`epoch 23 = 0.00189157` בשתי הריצות).

## תוצאות L3-SF ושער A

ה־post-processing הוקפא לפני test. תוצאת שער A היא **`STRONG_PASS`**.

| seed | Precision@4 | Recall@4 | F1@4 | F1@2 | F1@6 |
|---:|---:|---:|---:|---:|---:|
| 40401 | 0.9916 | 0.9655 | 0.9784 | 0.9773 | 0.9784 |
| 40402 | 0.9918 | 0.9639 | 0.9776 | 0.9767 | 0.9776 |
| 40403 | 0.9889 | 0.9636 | 0.9761 | 0.9751 | 0.9761 |

| seed | mean localization error | false positives לתמונה | mean recall לתמונה | predictions | GT |
|---:|---:|---:|---:|---:|---:|
| 40401 | 0.125 px | 3.44 | 0.9659 | 61511 | 63172 |
| 40402 | 0.128 px | 3.37 | 0.9644 | 61396 | 63172 |
| 40403 | 0.132 px | 4.55 | 0.9641 | 61557 | 63172 |

F1@4 לפי pattern (seed 40401): left_loop=0.9754, plain_arch=0.9789, right_loop=0.9791, tented_arch=0.9721, whorl=0.9783.

annotations בטווח 8 px משולי התמונה דווחו בנפרד ולא הוסרו מן המדד הראשי: seed 40401 recall=0.8882 (3664/4125), seed 40402 recall=0.8691 (3585/4125), seed 40403 recall=0.8882 (3664/4125).

פירוט לכל תמונה, לכל seed, נמצא ב־`artifacts/experiment-004/test_per_image.csv` (450 שורות: 150 תמונות test × 3 seeds).

Median seed F1@4 היה `0.9776`; precision `0.9916`; recall `0.9639`. ה־baseline הקפוא קיבל F1@4 של `0.7788`, ולכן היתרון המוחלט היה `0.1988`. פער ה־seeds היה `0.0023`.

Median F1 לפי run: R1=0.9786, R2=0.9772, R3=0.9785, R4=0.9779, R5=0.9760.

פרמטרי baseline: `{'nms_radius_px': 4, 'percentile': 99.25, 'sigma': 2.0}`.

## SD300

### נרמול ridge-scale ו־registration

| ppi | תמונות תקינות | UNRELIABLE_DISPERSION | too few tiles | median period | median factor | בתוך הגבול הקפוא | mated VALID |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1000 | 7/60 | 53 | 0 | 25.0 px | 1.36 | 5/60 | 0/20 |
| 2000 | 60/60 | 0 | 0 | 46.0 px | 0.74 | 60/60 | 2/20 |

אומדן ה־ridge period הקפוא מחפש lags בטווח `5-64 px`. ב־1000 ppi ה־period האמיתי הוא כ־16-22 px וההרמוניה שלו כ־40-60 px נופלת גם היא בתוך הטווח, ולכן אומדני ה־tiles יוצאים דו־מודליים ו־`MAD/median` חורג מן הסף הקפוא `0.25`. ב־2000 ppi ה־period הוא כ־40-50 px וההרמוניה שלו מחוץ לטווח החיפוש, ולכן האומדן יציב. זהו הגורם היחיד שמנע registrations תקפים ברזולוציה הראשית.

ב־1000 ppi היו `0` registrations תקפים מתוך 20 mated ו־`0` non-mated. Median Delta הראשי היה `NA` על `0` fingers paired-valid.

Bootstrap 95% CI: `[NA, NA]`.

Gate B: **`TRANSFER_INCONCLUSIVE`**. Experiment 001 נפתח רק לאחר freeze של 1000 ו־2000 ppi; ב־LEVEL3_USABLE נמצאו `0` מתוך `11` עם Delta חיובי.

ברזולוציה הראשית נפסלו `53` מתוך `60` התמונות משום שאומדן ה־ridge period שלהן לא היה אמין, ורק `7` קיבלו אומדן אמין. מתוכן `5` היו בתוך הגבול הקפוא `[0.2, 1.5]` ו־`2` מחוצה לו. כלומר הגורם החוסם היה כלל האמינות, ולא גבול ה־scale factor. בשום זוג לא עברו שתי התמונות יחד, ולכן כל 40 ה־registrations ב־1000 ppi הם `INVALID` עם `PREPROCESSING_FAILURE`.

ב־2000 ppi, שבו כל התמונות עברו preprocessing, ה־registration של `plain ↔ roll` הגיע ל־`VALID` ב־`2` מתוך 20 זוגות mated, ול־`VALID` ב־`0` מתוך 20 זוגות non-mated. בקרת ה־non-mated דורשת registration תקף בין שתי אצבעות שונות — בדיוק מה שכלל התקפות נועד לדחות — ולכן `Δ` אינו מוגדר גם ברזולוציה הזו. זו מגבלה מבנית של הפרוטוקול הקפוא, לא של ה־detector.

### בקרת density

ב־2000 ppi ה־detector הקפוא החזיר חציון של `1237` detections ב־plain ו־`3423` ב־roll, כלומר `849` detections למגה־פיקסל ו־`1325` למגה־פיקסל של שטח רכסים מוערך. לשם השוואה, ב־L3-SF יש כ־413 pore annotations לתמונת 512×512, שהם כ־1,570 למגה־פיקסל. כלומר ה־density אינו תוצר של detector שמציף את התמונה בנקודות.

בדיקת sensitivity של אותו impression ב־1000↔2000 נתנה median repeatability `0.7944` על `18` scores תקפים (בתוך הגבול הקפוא: `0.4161` על `12`); היא אינה נספרת כאוכלוסייה נוספת.

### variant exploratory של גבול ה־scale

ה־contingency הוקפא ב־`2026-08-29T17:15:26.686750+00:00`, לפני כל גישה ל־SD300 (`docs/experiments/004-scale-guard-contingency-amendment.md`). הוא אינו מכריע דבר: שער ב' וההכרעה הסופית חושבו מן הניתוח הראשי הקפוא בלבד.

תחת הכלל של המפרט (כישלון לפי אמינות האומדן בלבד, sanity guard `[0.2, 3.0]`) היו `0` registrations תקפים ב־mated, median Delta של `NA` על `0` fingers, ו־blinded outcome שהיה מתקבל: `TRANSFER_INCONCLUSIVE`.

## הכרעה וגבול הטענה

ההכרעה הסופית היא **`SYNTHETIC_LOCALIZATION_ONLY`**.

ה־detector הצליח ב־synthetic domain, אך אין evidence מספיק להעברה משכנעת ל־SD300; המשך מחקר חייב להתמקד ב־domain robustness לפני שימוש ב־pores ל־recognition.

אין להסיק מכאן precision/recall אנטומיים על SD300, generalization לכל sensor, שיפור recognition, matching של זהויות, synthetic-origin detection או liveness. הניסוי נעצר ללא matcher, descriptor או fusion.

## קבצי ראיה

המדדים וה־manifests הקומפקטיים נמצאים תחת `artifacts/experiment-004/`. weights, heatmaps, registrations ו־pixel-bearing derivatives נשארים local תחת `artifacts/experiment-004/local-large/` ומזוהים באמצעות SHA-256 ב־manifests.
