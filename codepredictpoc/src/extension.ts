import * as vscode from 'vscode';
import * as child_process from 'child_process';
import * as path from 'path';

let outputChannel : vscode.OutputChannel;

let CONTEXT_LENGTH = 64;

function runPythonPrediction(input: string): string[] {
    try {
		const extensionRoot = path.join(__dirname, '..')
		const pythonScriptPath = path.join(extensionRoot, 'src/py_scripts', 'client.py');
		let result = child_process.execSync(`python "${pythonScriptPath}" "${extensionRoot}" "pred" "${input}"`).toString();
		return result.split('\n').filter(line => line.trim() !== '');;;
    } catch (error) {
        console.error('Error running Python script:', error);
        return [];
    }
}  
  
function runFederatedLearning() {
	try {
		const pythonScriptPath = path.join(__dirname, '..', 'src/py_scripts', 'client.py');
		outputChannel.appendLine("Starting Learning!");
		let result = child_process.execSync(`python "${pythonScriptPath}" "train"`).toString();
		outputChannel.appendLine("Learning Successful!");
	} catch (error) {
		console.error('Error running Python script:', error);
		return [];
	}
}

export function activate(context: vscode.ExtensionContext) {
	outputChannel = vscode.window.createOutputChannel("federated-learning-result")
	let autoComplete = vscode.languages.registerCompletionItemProvider('python', {
        provideCompletionItems(document: vscode.TextDocument, position: vscode.Position, token: vscode.CancellationToken, context: vscode.CompletionContext) {
            const currentOffset = document.offsetAt(position);
            const startOffset = 0;
            const range = new vscode.Range(document.positionAt(startOffset), position);

            const currentWord = document.getText(range);

            const pythonOutput = runPythonPrediction(currentWord);

            const completionItems: vscode.CompletionItem[] = pythonOutput.map(item => {
                return new vscode.CompletionItem(item, vscode.CompletionItemKind.Text);
            });

            return completionItems;
        }
    }, ` `);

	let learner = vscode.commands.registerCommand("codepredictpoc.federated_learn", function() {
		runFederatedLearning();
	});

	console.log('Congratulations, your extension "codepredictpoc" is now active!');

	context.subscriptions.push(autoComplete);
	context.subscriptions.push(learner);
}

export function deactivate() {}
