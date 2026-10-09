/** Build an eight-slide editable Midterm deck from verified saved evidence.
 * Requires the bundled @oai/artifact-tool and presentation finalizer.
 * Usage: node build_midterm_slides.mjs --summary summary.json --build-dir <private>
 *   --output <new.pptx> --workspace-dir <task> --skill-dir <presentations skill>
 *   --modules-dir <bundled node_modules> --python <bundled python>
 * The caller records the artifact-operation marker once before authoring.
 */
import fs from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

function argsOf(argv) {
  const result = {};
  for (let i = 0; i < argv.length; i += 2) {
    if (!argv[i].startsWith('--') || !argv[i + 1]) throw new Error('Use --name value arguments');
    result[argv[i].slice(2)] = argv[i + 1];
  }
  for (const key of ['summary', 'build-dir', 'output', 'workspace-dir', 'skill-dir', 'modules-dir', 'python']) {
    if (!result[key]) throw new Error(`Missing --${key}`);
    result[key] = path.resolve(result[key]);
  }
  return result;
}
const opt = argsOf(process.argv.slice(2));
process.env.RUNTIME_NODE_MODULES = opt['modules-dir'];
process.env.RUNTIME_NODE = process.execPath;
process.env.RUNTIME_PYTHON = opt.python;
const summary = JSON.parse(await fs.readFile(opt.summary, 'utf8'));
if (summary.schema_version !== '1.0') throw new Error('Unsupported slide-summary schema');
if (summary.eda.scope !== 'training rows only') throw new Error('EDA must use training rows only');
if (summary.metrics.length !== 4 || new Set(summary.metrics.map(x => x.model)).size !== 4) throw new Error('Four distinct saved model rows are required');
for (const row of summary.metrics) for (const key of ['cv_mae_mean', 'cv_mae_std', 'test_mae', 'test_rmse', 'test_r2']) {
  if (!Number.isFinite(row[key])) throw new Error(`Non-finite ${row.model}/${key}`);
}
await fs.mkdir(opt['build-dir'], { recursive: true });
await fs.mkdir(path.dirname(opt.output), { recursive: true });
const { Presentation, PresentationFile, FileBlob } = await import(pathToFileURL(path.join(opt['modules-dir'], '@oai/artifact-tool/dist/artifact_tool.mjs')).href);
const utils = await import(pathToFileURL(path.join(opt['skill-dir'], 'container_tools/artifact_tool_utils.mjs')).href);
const FONT = utils.resolvePresentationFont({ fontFamily: opt.font || 'Arial' });
const C = { ink: '#102238', muted: '#526579', blue: '#087D9A', pale: '#EDF4F7', grid: '#DAE3E8', white: '#FFFFFF' };
const W = 1280, H = 720;
const pres = Presentation.create({ slideSize: { width: W, height: H } });
const D = summary.dataset, S = summary.split;
const names = { dummy: 'Median baseline', linear: 'Linear regression', knn: 'KNN regression', tree: 'Decision tree' };
const selected = summary.metrics.find(r => r.model === summary.selected_model);
const dummy = summary.metrics.find(r => r.model === 'dummy');
const nfmt = n => Number(n).toLocaleString('en-US');
const fixed = (n, p = 3) => Number(n).toFixed(p);
const shortClass = name => name.replace('Small Sport Utility Vehicle', 'Small SUV').replace('Standard Sport Utility Vehicle', 'Standard SUV').replace('Station Wagons', 'wagons').replace(' Cars', ' cars');
const smoke = Boolean(D.development_small_dataset);
const footer = smoke ? `Development sample, below 1,000 configurations` : `SE-2421   EPA combined consumption   ${D.snapshot_id}`;

function txt(slide, name, text, x, y, width, height, size = 27, { bold = false, color = C.ink, align = 'left' } = {}) {
  const shape = slide.shapes.add({ geometry: 'textbox', name,
    position: { left: x, top: y, width, height }, fill: 'none',
    line: { fill: 'none', width: 0, style: 'solid' } });
  shape.text = text;
  shape.text.style = { typeface: FONT, fontSize: size, color, bold, alignment: align,
    verticalAlignment: 'top', autoFit: 'none', wrap: 'square', insets: 0 };
  return shape;
}
function slide(title) {
  const s = pres.slides.add();
  s.background.fill = C.white;
  txt(s, 'slide-title', title, 64, 44, 1152, 76, 44, { bold: true });
  txt(s, 'footer', footer, 64, 680, 1092, 24, 16, { color: C.muted });
  txt(s, 'page-number', `${pres.slides.items.length}/8`, 1174, 680, 44, 24, 16, { color: C.muted, align: 'right' });
  return s;
}
function notes(s, seconds, narrative, sourceNames = []) {
  s.speakerNotes.text = `Suggested timing: ${seconds} seconds.\n\n${narrative}\n\nSources: ${sourceNames.join('\n')}`;
}
function table(s, values, x, y, width, height, widths, fontSize = 24) {
  const t = s.tables.add({ rows: values.length, columns: values[0].length,
    left: x, top: y, width, height, columnWidths: widths, values });
  t.borders.assign({ fill: C.grid, width: 1, style: 'solid' });
  for (let row = 0; row < values.length; row++) for (let col = 0; col < values[0].length; col++) {
    const cell = t.getCell(row, col);
    cell.fill = row === 0 ? C.ink : C.white;
    cell.text.style = { typeface: FONT, fontSize, color: row === 0 ? C.white : C.ink,
      bold: row === 0, verticalAlignment: 'middle', insets: 10 };
  }
  return t;
}
function bar(s, title, categories, values, x, y, width, height, { unit = 'L/100 km', color = C.blue, horizontal = false } = {}) {
  const chart = s.charts.add('bar', { title, titlePlacement: 'aboveChart',
    position: { left: x, top: y, width, height }, categories,
    series: [{ name: unit, values, fill: color, valuesFormatCode: '0.00' }],
    barOptions: { direction: horizontal ? 'bar' : 'column', grouping: 'clustered', gapWidth: 65 },
    hasLegend: false, chartFill: C.white, plotAreaFill: C.white,
    titleTextStyle: { fontSize: 27, fill: C.ink, bold: true },
    xAxis: { textStyle: { fontSize: horizontal ? 21 : 23, fill: C.muted }, line: { fill: C.grid, width: 1 }, majorGridlines: null },
    yAxis: { min: 0, numberFormatCode: '0.0', textStyle: { fontSize: 23, fill: C.muted },
      title: { text: unit, textStyle: { fontSize: 23, fill: C.muted } },
      majorGridlines: { fill: C.grid, width: 1, style: 'solid' } },
    dataLabels: { showValue: true, position: 'outEnd', numberFormatCode: '0.00',
      textStyle: { fontSize: 23, fill: C.ink } },
  });
  utils.applyPresentationChartFont(chart, { fontFamily: FONT });
  return chart;
}

// 1. Research question and source
{
  const s = pres.slides.add();
  s.background.fill = C.ink;
  txt(s, 'cover-title', 'Predicting EPA combined\nfuel consumption', 64, 110, 1120, 190, 64, { bold: true, color: C.white });
  txt(s, 'cover-question', 'How accurately can technical specifications\npredict consumption for gasoline cars and SUVs?', 64, 335, 1090, 108, 32, { color: C.white });
  txt(s, 'cover-source', 'FuelEconomy.gov API   US configurations   2015вЂ“2025', 64, 500, 1100, 42, 26, { color: '#A9CDD7' });
  txt(s, 'cover-team', `${summary.project.group}\n${summary.project.members.join(' and ')}`, 64, 578, 1100, 68, 25, { color: C.white });
  if (smoke) txt(s, 'development-status', 'DEVELOPMENT SAMPLE', 64, 54, 1100, 32, 22, { bold: true, color: '#A9CDD7' });
  notes(s, 55, `Our project asks whether ordinary technical specifications can predict EPA estimated combined fuel consumption. The response is litres per 100 kilometres, which is easier to interpret as consumption than miles per gallon. The source describes vehicle configurations, not private people or individual sold cars. We focus on gasoline cars and SUVs from the US catalogue for model years 2015 through 2025. We collect individual records with our own Python code through the documented FuelEconomy.gov API. The practical use is comparing the expected consumption associated with different specifications. We evaluate predictive association rather than claiming that one characteristic causes a particular level of consumption. ${smoke ? 'This deck currently uses a development sample and does not meet the minimum benchmark volume.' : `The current benchmark contains ${nfmt(D.cleaned_rows)} retained configurations.`}`, ['https://www.fueleconomy.gov/feg/ws/index.shtml', 'docs/DATA_CONTRACT.md']);
}
// 2. Scope, collection and cleaning
{
  const s = slide('Collection and scope');
  txt(s, 'raw-count', `${nfmt(D.raw_records)} individual records`, 64, 165, 640, 65, 42, { bold: true });
  txt(s, 'excluded-count', `${nfmt(D.excluded_records)} scope or quality exclusions`, 64, 250, 640, 45, 28);
  txt(s, 'duplicate-count', `${nfmt(D.duplicates_removed)} confirmed duplicate records removed`, 64, 314, 640, 62, 28);
  txt(s, 'retained-count', `${nfmt(D.cleaned_rows)} retained configurations`, 64, 413, 650, 66, 42, { bold: true, color: C.blue });
  txt(s, 'scope', `${D.years.length} observed model years\n${D.manufacturers_n} manufacturers\nGasoline cars, wagons and SUVs\nHybrid and alternative fuels excluded`, 765, 167, 449, 214, 27);
  txt(s, 'source-preservation', 'Raw responses, UTC dates and SHA-256\nResumable menus and record requests', 64, 535, 700, 78, 26, { color: C.muted });
  txt(s, 'target-formula', 'Target in L/100 km\n235.2145833333333 / comb08', 765, 447, 449, 105, 28, { bold: true });
  txt(s, 'sampling-status', D.full_catalogue_complete ? 'Completed catalogue traversal within the declared scope' : 'Bounded catalogue sample with unequal inclusion probabilities', 64, 635, 1136, 34, 23, { color: C.muted });
  notes(s, 70, `The collection retrieves year, manufacturer, model and option menus before requesting each individual vehicle record. We preserve raw bytes, dates and checksums so another person can inspect the source and reproduce cleaning. The downloaded count and cleaned count are different quantities. We exclude unsupported fuel types, secondary fuels, hybrids including mild hybrids, classes outside our declared cars and SUV scope, invalid targets and insufficient or invalid engine specifications. Start-stop alone is not a hybrid indicator. We keep missing values for individual optional predictors, because the training pipeline can impute them. We form duplicate candidates without using the target and remove only confirmed duplicate configurations. ${nfmt(D.raw_records)} records produce ${nfmt(D.cleaned_rows)} retained rows after ${nfmt(D.excluded_records)} exclusions and ${nfmt(D.duplicates_removed)} duplicate removals. ${D.full_catalogue_complete ? 'The recorded snapshot reports completed catalogue traversal.' : 'This bounded sample is not a completed census and does not estimate the mix of vehicles sold.'} EPA comb08 uses US gallons. We convert it with a fixed factor and never impute the target or silently replace it with comb08U.`, ['data/interim/cleaned_summary.json', 'data/interim/cleaning_summary.csv', 'raw snapshot.json', 'docs/DATA_CONTRACT.md']);
}
// 3. EDA from train only
{
  const s = slide('Training data exploration');
  txt(s, 'eda-scope', `Training rows only, n = ${nfmt(S.n_train)}. Means describe configurations in this sample.`, 64, 126, 1136, 43, 26, { color: C.muted });
  const e = summary.eda.engine_bin_target, c = summary.eda.class_family_target;
  bar(s, 'Consumption by engine size', e.map(r => `${r.label}\nn=${r.n}`), e.map(r => r.mean_target), 64, 186, 550, 390);
  bar(s, 'Consumption by class family', c.map(r => `${r.label}\nn=${r.n}`), c.map(r => r.mean_target), 664, 186, 550, 390);
  txt(s, 'eda-interpretation', 'Engine size and vehicle class provide predictive context.\nGroup means and correlations do not establish causal effects.', 64, 604, 1148, 65, 26);
  notes(s, 65, `These editable charts use only the training rows from the saved split. The left chart groups engine displacement into predeclared intervals of zero to two litres, over two to three, over three to four and over four litres. Labels show the number of observations supporting each mean. The right chart combines the API vehicle classes into cars, station wagons and SUVs for readable comparison. The notebook retains the detailed class breakdown and target distribution. We describe these observed differences as predictive associations. We do not remove inconvenient high-consumption vehicles or change bins after looking at the test errors. Configuration counts give no information about sales volume. The number of rows in a group also matters because a mean based on a few rare vehicles may be unstable. The plots motivate comparison of a linear model with methods that can capture local patterns and nonlinear thresholds.`, ['frozen split_manifest.csv', 'training rows of vehicles.parquet', 'notebooks/01_midterm.ipynb', 'docs/EXPERIMENT_PROTOCOL.md']);
}
// 4. Safe features and preprocessing
{
  const s = slide('Predictors and preprocessing');
  txt(s, 'feature-heading', 'Seven structured predictors', 64, 157, 600, 48, 32, { bold: true });
  txt(s, 'numeric-features', 'Numeric\nModel year, displacement, cylinders', 64, 237, 608, 87, 28);
  txt(s, 'categorical-features', 'Categorical\nManufacturer, transmission, drivetrain, class', 64, 373, 608, 124, 28);
  txt(s, 'pipeline-heading', 'Fit inside each training fold', 758, 157, 457, 58, 32, { bold: true });
  txt(s, 'pipeline-body', 'Numeric median imputation\nMissing indicators and scaling\n\nUnknown categorical values\nOne-hot encoding\n\nSaved preprocessing plus estimator', 758, 236, 460, 298, 27);
  txt(s, 'leakage-rule', 'Other fuel-economy values, costs, emissions and scores stay outside X.\nModel names and engine text remain reserved for Final experiments.', 64, 588, 1149, 80, 26, { color: C.muted });
  notes(s, 60, `All four Midterm models use the same seven predictors. Model year, displacement and cylinder count are numeric. Manufacturer, transmission, drivetrain and vehicle class are categorical. A strict allowlist prevents the estimator from seeing any other raw columns. We specifically exclude other fuel economy values, fuel costs, emissions and efficiency scores because they can reveal the answer. Identifiers, base model, source dates and group labels also stay outside the predictors. We preserve model names and engine description for Final, but the current models do not use them. Preprocessing is part of the saved pipeline. Within each training fold, we fit numeric median imputation, missing indicators and scaling, then categorical imputation and one-hot encoding with unknown categories allowed. Validation and test rows use those already fitted transformations. This prevents information from validation or test rows from influencing imputation or encoding.`, ['src/fuel_consumption/features.py', 'configs/project.json', 'docs/EXPERIMENT_PROTOCOL.md']);
}
// 5. Group split, CV and fixed models
{
  const s = slide('Family split and regression models');
  txt(s, 'split-counts', `${nfmt(S.n_train)} train rows\n${nfmt(S.n_test)} test rows`, 64, 168, 430, 110, 37, { bold: true });
  txt(s, 'split-groups', `${S.n_train_groups} train families\n${S.n_test_groups} test families\n\n20% of groups for test, seed 42\nFive training GroupKFold folds`, 64, 320, 449, 203, 27);
  txt(s, 'group-definition', 'Manufacturer plus base model\nOne family stays together across years', 64, 563, 450, 85, 25, { color: C.muted });
  table(s, [['Model', 'Fixed starting parameters'],
    ['Dummy', 'Training-target median'], ['Linear regression', 'Intercept enabled'],
    ['KNN regression', 'k = 15, distance weights, p = 2'],
    ['Decision tree', 'Depth 8, leaf minimum 10, seed 42']],
    535, 171, 680, 337, [238, 442], 24);
  txt(s, 'selection-protocol', 'Selection uses mean training CV MAE.\nThe test split stays fixed across course stages.', 535, 554, 680, 88, 27);
  notes(s, 75, `We evaluate transfer to vehicle families that do not appear in training. A family combines manufacturer and base model, with audited full model-name fallbacks when necessary. We keep a family together across model years. The fixed GroupShuffleSplit assigns twenty percent of groups to test with seed forty-two, so the test share of rows need not equal twenty percent. The saved manifest contains ${nfmt(S.n_train)} training rows in ${S.n_train_groups} families and ${nfmt(S.n_test)} test rows in ${S.n_test_groups} families. Five grouped cross-validation folds use training data only, with zero family overlap between fitting and validation. We do not search for a seed with convenient results. The four models include a median dummy baseline and three regressors confirmed in the supplied Week Three and Four lectures. We choose the model by mean CV MAE before calculating test predictions. These are fixed starting parameters rather than tuned optima. Endterm tuning will use grouped training CV, while the course keeps the same test split for comparison.`, ['split_metadata.json', 'group_alias_audit.csv', 'references/Lecture_3.pdf, PDF pages 28, 32, 35вЂ“37', 'references/Lecture_4.pdf, PDF pages 15, 25, 28, 40', 'docs/EXPERIMENT_PROTOCOL.md']);
}
// 6. Consistent metrics for all models
{
  const s = slide('Cross-validation and test results');
  table(s, [['Model', 'CV MAE', 'CV std', 'Test MAE', 'Test RMSE', 'Test RВІ'],
    ...summary.metrics.map(r => [names[r.model], fixed(r.cv_mae_mean), fixed(r.cv_mae_std), fixed(r.test_mae), fixed(r.test_rmse), fixed(r.test_r2)])],
    64, 157, 1152, 292, [282, 174, 174, 174, 174, 174], 24);
  bar(s, 'Test MAE', ['Dummy', 'Linear', 'KNN', 'Tree'], summary.metrics.map(r => r.test_mae), 64, 477, 560, 178);
  txt(s, 'cv-winner', `${names[selected.model]} wins by CV MAE`, 700, 488, 506, 90, 31, { bold: true, color: C.blue });
  txt(s, 'test-improvement', `${fixed(100 * (dummy.test_mae - selected.test_mae) / dummy.test_mae, 1)}% lower test MAE than the median baseline`, 700, 588, 506, 66, 26);
  notes(s, 75, `The table reports every predefined model on the same data and split. MAE and RMSE use litres per one hundred kilometres, and lower values indicate smaller errors. CV MAE is the mean of five positive fold values. The reported standard deviation describes their spread and is not a confidence interval. R-squared is a relative explanatory measure and is not a percentage of correctly predicted vehicles. The selected model is ${names[selected.model]}, based on its saved training CV MAE of ${fixed(selected.cv_mae_mean)} with standard deviation ${fixed(selected.cv_mae_std)}. Its test MAE is ${fixed(selected.test_mae)}, compared with ${fixed(dummy.test_mae)} for the median baseline. That difference corresponds to ${fixed(100 * (dummy.test_mae - selected.test_mae) / dummy.test_mae, 1)} percent lower test MAE. We did not use the test values to change the model selection or hyperparameters. ${smoke ? 'These values establish only that the workflow runs on the development sample.' : 'The results apply to this bounded benchmark and held-out families under the declared protocol.'}`, ['saved metrics.csv', 'saved cv_selection.json', 'saved fold_metrics.csv', 'docs/EXPERIMENT_PROTOCOL.md']);
}
// 7. Test subgroup diagnostics and real large error
{
  const s = slide('Errors by class and engine size');
  const groups = summary.subgroups.filter(r => r.split === 'test');
  const classes = groups.filter(r => r.subgroup_field === 'vehicle_class').sort((a, b) => b.mae - a.mae).slice(0, 5);
  table(s, [['Class, largest MAE groups', 'n', 'MAE', 'Bias'], ...classes.map(r => [shortClass(r.subgroup), String(r.n), fixed(r.mae, 2), fixed(r.bias, 2)])],
    64, 169, 672, 320, [391, 71, 105, 105], 23);
  const engines = groups.filter(r => r.subgroup_field === 'engine_size_bin' && !r.subgroup.toLowerCase().includes('missing')).sort((a, b) => a.subgroup.localeCompare(b.subgroup));
  bar(s, 'Test MAE by engine size', engines.map(r => `${r.subgroup.replace(' L', '')}\nn=${r.n}`), engines.map(r => r.mae), 780, 174, 435, 315);
  txt(s, 'subgroup-support', `Selected model: ${names[selected.model]}. Groups with n < 30 have limited support.\nBias = predicted consumption minus actual consumption.`, 64, 529, 1145, 72, 25, { color: C.muted });
  const err = summary.large_errors[0];
  if (err) txt(s, 'large-error', `Largest test error: ${err.model_year} ${err.manufacturer} ${err.model_name}, ID ${err.vehicle_id}\nActual ${fixed(err.y_true, 2)}, predicted ${fixed(err.y_pred, 2)}, absolute error ${fixed(err.abs_error, 2)} L/100 km`, 64, 611, 1145, 62, 24);
  notes(s, 80, `Overall MAE can hide different errors across vehicle classes and engine sizes. These are descriptive test diagnostics for the model chosen earlier by training CV. The table shows up to five classes with the largest test MAE and preserves their observation counts. The chart shows test MAE in the predeclared displacement intervals. We flag groups with fewer than thirty observations as having limited support and avoid claiming a stable ranking of reliability from a small group. Bias means prediction minus actual, so positive bias indicates that the model overestimates consumption. ${err ? `One real large-error example is the ${err.model_year} ${err.manufacturer} ${err.model_name}, source ID ${err.vehicle_id}. Its actual converted EPA consumption is ${fixed(err.y_true, 2)}, the prediction is ${fixed(err.y_pred, 2)}, and the absolute error is ${fixed(err.abs_error, 2)} litres per one hundred kilometres.` : 'The saved large-error report contains the source configurations for further inspection.'} Missing predictors such as vehicle mass or power may explain some differences, but this is a hypothesis. Endterm choices must use training out-of-fold diagnostics rather than tuning to these test examples.`, ['saved subgroup_errors.csv', 'saved large_errors.csv', 'docs/EXPERIMENT_PROTOCOL.md']);
}
// 8. Evidence-based conclusion, course roadmap and attribution
{
  const s = slide('Conclusion and next stages');
  txt(s, 'main-conclusion', `${names[selected.model]}: test MAE ${fixed(selected.test_mae)} L/100 km`, 64, 163, 1150, 72, 35, { bold: true, color: C.blue });
  txt(s, 'limits', 'Limits\nEPA estimates and rounded target values\nBounded configurations, no sales weighting\nUnseen families rather than future-year forecasting', 64, 281, 655, 191, 27);
  txt(s, 'roadmap', 'Endterm\nEnsemble tuning, clustering, PCA and a neural network\n\nFinal\nModel and engine text comparison, local prediction app', 767, 279, 449, 238, 27);
  txt(s, 'attribution', 'AI demonstration; simulated team roles:\nNikita: data and collection. Bakyt: models, analysis and slides.', 64, 559, 1150, 83, 26, { color: C.muted });
  notes(s, 80, `The selected model achieves a test MAE of ${fixed(selected.test_mae)} litres per one hundred kilometres on this saved benchmark. It improves on the median baseline, but we should interpret that result within the collection and evaluation limits. EPA estimates differ from an individual driverвЂ™s realised consumption, and the original combined miles-per-gallon field is rounded. The bounded catalogue sample has unequal inclusion probabilities and no sales weighting. The grouped protocol examines unseen vehicle families within the sampled years rather than predicting future model years. ${D.unresolved_duplicates ? `${D.unresolved_duplicates} duplicate candidate groups remain an additional documented limitation.` : ''} At Endterm we plan ensemble tuning through grouped training validation, descriptive clustering and dimensionality reduction, and a neural network. At Final we plan controlled comparisons of structured features with model names and engine description text on the same split, followed by a local prediction application. Codex assisted with code, collection and analysis artifacts, documentation and presentation drafts. This is an AI capability demonstration rather than an actual course submission. The user requested simulated team roles: Nikita owns collection and data preparation; Bakyt owns modeling, analysis and presentation. The automated implementation is separately recorded in the AI log.`, ['docs/DATA_CARD.md', 'docs/CONTRIBUTIONS.md', 'docs/PROJECT_PLAN.md', 'references/Project_Guide.pdf']);
}

if (pres.slides.items.length !== 8) throw new Error('Expected exactly eight slides');
const draft = path.join(opt['build-dir'], 'candidate.pptx');
await (await PresentationFile.exportPptx(pres)).save(draft);
await fs.writeFile(path.join(opt['build-dir'], 'source_summary.json'), JSON.stringify(summary, null, 2));
const requirement = { explicitTotalSlideCount: 8, requiredNativeTableOwnerSlides: [5, 6, 7],
  requiredNativeChartOwnerSlides: [3, 6, 7], materializeLiteralChartWorkbooks: true };
const result = await utils.finalizePresentation({ ...requirement,
  workspaceDir: opt['workspace-dir'], candidatePath: draft, finalPath: opt.output,
  pythonExecutable: opt.python,
  integrityValidatorPath: path.join(opt['skill-dir'], 'container_tools/inspect_presentation_package_integrity.py'),
  layoutValidatorPath: path.join(opt['skill-dir'], 'container_tools/inspect_presentation_layout_geometry.py'),
  layoutArgs: ['--expected-slide-size-emu', '12192000,6858000', '--validate-bullet-geometry', '--validate-heading-fit',
    ...requirement.requiredNativeTableOwnerSlides.flatMap(number => ['--require-native-table-slide', String(number)])],
  fontPolicy: { basis: 'design', families: [FONT] }, verifyArtifactToolImport: true,
  receiptPath: path.join(opt['build-dir'], `${path.basename(opt.output)}.validation.json`),
});
const finalPres = await PresentationFile.importPptx(await FileBlob.load(opt.output));
for (const [i, s] of finalPres.slides.items.entries()) {
  const stem = `slide-${String(i + 1).padStart(2, '0')}`;
  const png = await finalPres.export({ slide: s, format: 'png', scale: 1 });
  await fs.writeFile(path.join(opt['build-dir'], `${stem}.png`), new Uint8Array(await png.arrayBuffer()));
  const layout = await s.export({ format: 'layout' });
  await fs.writeFile(path.join(opt['build-dir'], `${stem}.layout.json`), await layout.text());
}
await fs.writeFile(path.join(opt['build-dir'], 'build_receipt.json'), JSON.stringify({ final_path: opt.output, slide_count: 8,
  source_summary: opt.summary, validation: result, visual_review_required: true, smoke }, null, 2));
console.log(JSON.stringify({ final_path: opt.output, previews: opt['build-dir'], slide_count: 8, smoke }));
