# `Node.js`

<h2>Table of contents</h2>

- [What is `Node.js`](#what-is-nodejs)
- [`nvm`](#nvm)
  - [Install `nvm`](#install-nvm)
- [Install `Node.js`](#install-nodejs)
  - [Install `Node.js` using `nvm`](#install-nodejs-using-nvm)
  - [Install `Node.js` using the commands from the official site](#install-nodejs-using-the-commands-from-the-official-site)
- [Check that `Node.js` works](#check-that-nodejs-works)
- [`npm`](#npm)
  - [`package.json`](#packagejson)
  - [`node_modules`](#node_modules)
- [Common `npm` commands](#common-npm-commands)
  - [`npm install`](#npm-install)
- [Common `npm` actions](#common-npm-actions)
  - [Install `Node.js` dependencies in the directory](#install-nodejs-dependencies-in-the-directory)

## What is `Node.js`

`Node.js` is a runtime environment that executes `JavaScript` outside of a browser. In this project, it is used to run the frontend development server and build tools.

Docs:

- [Node.js documentation](https://nodejs.org/en/docs)

## `nvm`

`nvm` (Node Version Manager) is a tool for installing and switching between multiple versions of [`Node.js`](#what-is-nodejs).

See [Install `nvm`](#install-nvm).

Docs:

- [`nvm` repository](https://github.com/nvm-sh/nvm)

### Install `nvm`

1. Copy the single-line script from the [installation instructions](https://github.com/nvm-sh/nvm#installing-and-updating).

2. [Run the copied script in the `VS Code Terminal`](./vs-code.md#run-a-command-in-the-vs-code-terminal).

3. To check that [`nvm`](#nvm) is installed,

   [run in the `VS Code Terminal`](./vs-code.md#run-a-command-in-the-vs-code-terminal):

   ```terminal
   nvm --version
   ```

   The output should be similar to this:

   ```terminal
   0.40.3
   ```

## Install `Node.js`

- Method 1: [Install `Node.js` using `nvm`](#install-nodejs-using-nvm)
- Method 2: [Install `Node.js` using the commands from the official site](#install-nodejs-using-the-commands-from-the-official-site)

### Install `Node.js` using `nvm`

1. [Install `nvm`](#install-nvm) if not yet installed.

2. To install [`Node.js`](#what-is-nodejs) using [`nvm`](#nvm),

   [run in the `VS Code Terminal`](./vs-code.md#run-a-command-in-the-vs-code-terminal):

   ```terminal
   nvm install 25.7.0
   ```

3. The output should be similar to this:

   ```terminal
   Downloading and installing node v25.7.0...
   Now using node v25.7.0 (npm v11.10.1)
   ```

4. To set this version as the default,

   [run in the `VS Code Terminal`](./vs-code.md#run-a-command-in-the-vs-code-terminal):

   ```terminal
   nvm alias default node
   ```

5. [Check that `Node.js` works](#check-that-nodejs-works).

### Install `Node.js` using the commands from the official site

1. Go to the [`Download Node.js` page](https://nodejs.org/en/download)

2. Configure the instructions for a [terminal](./terminal.md). If you use:

   - `Linux`: Get `v25.x.x` for `Linux` using `<tool>` with `<package-manager>`
   - `macOS`: Get `v25.x.x` for `macOS` using `<tool>` with `<package-manager>`
   - `Windows`: Get `v25.x.x` for `Linux` using `<tool>` with `<package-manager>`

   Choose `<tool>` and `<package-manager>` that you like.
  
   We recommend to replace:

   - `<tool>` with `nvm`
   - `<package-manager>` with `npm`

3. To copy the instructions to clipboard,

   Click `Copy to clipboard`.

4. [Run the copied commands in the `VS Code Terminal`](./vs-code.md#run-a-command-in-the-vs-code-terminal).

## Check that `Node.js` works

1. [Check the current shell in the `VS Code Terminal`](./vs-code.md#check-the-current-shell-in-the-vs-code-terminal).
2. To check the `Node.js` version,

   [run in the `VS Code Terminal`](./vs-code.md#run-a-command-in-the-vs-code-terminal):

   ```terminal
   node --version
   ```

3. The output should be similar to this:

   ```terminal
   v25.7.0
   ```

<!-- TODO pnpm -->
<!-- TODO install npm with nix because when installing on the VM using nvm libatomic is missing -->

## `npm`

`npm` is the default package manager for [`Node.js`](#what-is-nodejs).
It installs and manages project dependencies declared in [`package.json`](#packagejson).

It is installed automatically when you install [`Node.js`](#install-nodejs).

Docs:

- [`npm` documentation](https://docs.npmjs.com/)

### `package.json`

`package.json` is a configuration [file](./file-system.md#file) in a [`Node.js`](#what-is-nodejs) project that declares the project's [dependencies](./package-manager.md#dependency), scripts, and metadata.

[`npm`](#npm) reads it to know which packages to install and which commands to run.

### `node_modules`

`node_modules` stores all [`Node.js`](#nodejs) modules installed using [`npm`](#npm) or another package manager for `Node.js`.

This [directory](./file-system.md#directory) is [`.gitignore`](./git.md#gitignore)-d.

## Common `npm` commands

- [`npm install`](#npm-install)

### `npm install`

This command [installs packages in the specified directory](#install-nodejs-dependencies-in-the-directory).

Executes postinstall hooks.

## Common `npm` actions

- [Install `Node.js` dependencies in the directory](#install-nodejs-dependencies-in-the-directory)

### Install `Node.js` dependencies in the directory

> [!NOTE]
> See [`npm install`](#npm-install), [`package.json`](#packagejson), [directory](./file-system.md#directory).

1. [Open in `VS Code` the project directory](./vs-code.md#open-the-directory).

2. If the `package.json` is in the [root directory of the repository](./git.md#root-directory-of-the-repository),

   [run in the `VS Code Terminal`](./vs-code.md#run-a-command-in-the-vs-code-terminal):

   ```terminal
   npm install
   ```

3. If the `package.json` is in a [subdirectory](./file-system.md#subdirectory) of the root directory of the repository,

   run in the `VS Code Terminal`:

   ```
   npm install --prefix <frontend-dir>
   ```

   Example:

   ```
   npm install --prefix frontend
   ```

4. Verify that the output is similar to this:

   ```terminal
   added 143 packages, and audited 144 packages in 3s
   ```

5. [Open the `VS Code Explorer`](./vs-code.md#vs-code-explorer).

6. Verify that there is the `node_modules` directory in the directory where you wanted to install the dependencies.
