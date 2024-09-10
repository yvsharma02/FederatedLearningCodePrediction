import * as vscode from 'vscode';
import * as child_process from 'child_process';
import * as path from 'path';
import * as fs from 'fs';

let outputChannel : vscode.OutputChannel;

let CONTEXT_LENGTH = 64;


function listJavaFiles(dirPath: string): string {
    let javaFiles: string[] = [];

    // Helper function to recursively find .java files
    function findJavaFiles(currentPath: string) {
		console.log(currentPath)
        const files = fs.readdirSync(currentPath);

        for (const file of files) {
            const fullPath = path.join(currentPath, file);
            const stat = fs.statSync(fullPath);

            if (stat.isDirectory()) {
                // Recursively search in subfolders
                findJavaFiles(fullPath);
            } else if (path.extname(fullPath) === '.java') {
                // Add .java files to the array
                javaFiles.push(fullPath);
            }
        }
    }

    // Start the recursive search from the provided directory path
    findJavaFiles(dirPath);
	console.log(javaFiles.join('\n'));
    // Join the list of .java files with newline character
    return javaFiles.join('\n');
}

function runPythonPrediction(input: string): string[] {
    try {
		const extensionRoot = path.join(__dirname, '..')
		const pythonScriptPath = path.join(extensionRoot, 'src/py_scripts', 'client.py');
		let result = child_process.execSync(`python "${pythonScriptPath}" "${extensionRoot}" "pred" "${input}"`).toString();
		return result.split('\n').filter(line => line.trim() !== '');
    } catch (error) {
        console.error('Error running Python script:', error);
        return [];
    }
}  
  
function runFederatedLearning() {
	try {
		if (vscode.workspace.workspaceFolders !== undefined)  {
			let wf = vscode.workspace.workspaceFolders[0].uri.fsPath
			outputChannel.appendLine("Reading files: " + wf)
			let input = listJavaFiles(wf);
			outputChannel.appendLine("Read files!")

			const extensionRoot = path.join(__dirname, '..')
			const pythonScriptPath = path.join(extensionRoot, 'src/py_scripts', 'client.py');

			outputChannel.appendLine("Starting training")
			console.log(child_process.execSync(`python "${pythonScriptPath}" "${extensionRoot}" "train" "${input}"`))
			outputChannel.appendLine("Finishing training")
		}
	} catch (error) {
		console.error('Error running Python script:', error);
		return [];
	}
}

function readJavaFilesInDirAsString(dir: string): string {
    let allJavaFilesContent = '';

    // Get all files in the directory
    const files = fs.readdirSync(dir);

    // Loop through each file
    for (const file of files) {
        const filePath = path.join(dir, file);
        
        // Check if the current path is a file and has a .java extension
        if (fs.statSync(filePath).isFile() && path.extname(file) === '.java') {
            // Read the file content and append it with $EOF$
            const fileContent = fs.readFileSync(filePath, 'utf-8');
            allJavaFilesContent += fileContent + '<|endoftext|>';
        }
    }

    return allJavaFilesContent;
}

export function activate(context: vscode.ExtensionContext) {
	outputChannel = vscode.window.createOutputChannel("federated-learning-result")

	let autoComplete = vscode.languages.registerCompletionItemProvider('java', {
        provideCompletionItems(document: vscode.TextDocument, position: vscode.Position, token: vscode.CancellationToken, context: vscode.CompletionContext) {
            const startOffset = 0;
            const range = new vscode.Range(document.positionAt(startOffset), position);

            const ctx = document.getText(range);

            const pythonOutput = runPythonPrediction(ctx);

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
