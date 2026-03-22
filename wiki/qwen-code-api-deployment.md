# `Qwen Code` API deployment

<h2>Table of contents</h2>

- [About the `Qwen Code` API deployment](#about-the-qwen-code-api-deployment)
- [Deploy the `Qwen Code` API (REMOTE)](#deploy-the-qwen-code-api-remote)
  - [Clone the `Qwen Code` API repository (REMOTE)](#clone-the-qwen-code-api-repository-remote)
  - [Enter the `Qwen Code` API repository directory (REMOTE)](#enter-the-qwen-code-api-repository-directory-remote)
  - [Pull the latest changes from the `Qwen Code` API repository (REMOTE)](#pull-the-latest-changes-from-the-qwen-code-api-repository-remote)
  - [Prepare the environment in the `Qwen Code` API repository (REMOTE)](#prepare-the-environment-in-the-qwen-code-api-repository-remote)
  - [Start the `Qwen Code` API (REMOTE)](#start-the-qwen-code-api-remote)

## About the `Qwen Code` API deployment

This page describes how to deploy the [`Qwen Code` API](./qwen-code-api.md#what-is-qwen-code-api) on [your VM](./vm.md#your-vm) using [`Docker Compose`](./docker-compose.md#what-is-docker-compose).

## Deploy the `Qwen Code` API (REMOTE)

Complete these steps:

1. [Connect to the VM as the user `admin`](./vm-access.md#connect-to-the-vm-as-the-user-user-local).
2. [Set up the `Qwen Code` CLI (REMOTE)](./qwen-code.md#set-up-the-qwen-code-cli-remote).
3. [Clone the `Qwen Code` API repository (REMOTE)](#clone-the-qwen-code-api-repository-remote).
4. [Pull the latest changes from the `Qwen Code` API repository (REMOTE)](#pull-the-latest-changes-from-the-qwen-code-api-repository-remote).
5. [Enter the `Qwen Code` API repository directory (REMOTE)](#enter-the-qwen-code-api-repository-directory-remote).
6. [Prepare the environment in the `Qwen Code` API repository (REMOTE)](#prepare-the-environment-in-the-qwen-code-api-repository-remote).
7. [Start the `Qwen Code` API (REMOTE)](#start-the-qwen-code-api-remote).
8. [Check that the `Qwen Code` API is accessible](./qwen-code-api.md#check-that-the-qwen-code-api-is-accessible) on the VM (REMOTE).

### Clone the `Qwen Code` API repository (REMOTE)

1. [Clone the `Qwen Code` API repository](./git-vscode.md#clone-the-repository-using-the-vs-code-terminal).

   Replace the placeholders:

   - `<repo-url>` with `https://github.com/inno-se-toolkit/qwen-code-api`

   - `<repo-directory>` with `~/qwen-code-api`.

### Enter the `Qwen Code` API repository directory (REMOTE)

1. To enter the repository directory,

   [run in the `VS Code Terminal`](./vs-code.md#run-a-command-in-the-vs-code-terminal):

   ```terminal
   cd ~/qwen-code-api
   ```

### Pull the latest changes from the `Qwen Code` API repository (REMOTE)

1. To pull the latest changes,

   [run in the `VS Code Terminal`](./vs-code.md#run-a-command-in-the-vs-code-terminal):

   ```terminal
   git pull
   ```

   The output should be:

   ```terminal
   Already up to date.
   ```

2. [Hard reset the branch `main`](./git-vscode.md#hard-reset-the-branch-branch).

### Prepare the environment in the `Qwen Code` API repository (REMOTE)

1. To create [`qwen-code-api/.env.secret`](./qwen-code-api-dotenv-secret.md),

   [run in the `VS Code Terminal`](./vs-code.md#run-a-command-in-the-vs-code-terminal):

   ```terminal
   cp .env.example .env.secret
   ```

2. [Set the variable in `.env.secret`](./environments.md#set-the-variable-to-value-in-the-env-file-at-file-path):

   - [`QWEN_CODE_API_KEY`](./qwen-code-api-dotenv-secret.md#qwen_code_api_key)

### Start the `Qwen Code` API (REMOTE)

1. To start the `Qwen Code` API via [`Docker Compose`](./docker-compose.md#what-is-docker-compose),

   [run in the `VS Code Terminal`](./vs-code.md#run-a-command-in-the-vs-code-terminal):

   ```terminal
   docker compose --env-file .env.secret up --build -d
   ```

   > <h3>Troubleshooting</h3>
   >
   > [**Port conflict (`port is already allocated`)**](./docker.md#port-conflict-port-is-already-allocated)
   >
   > **`network lms-network was found but has incorrect label com.docker.compose.network set to "default" (expected: "lms-network")`**
   >
   > 1. [Clean up `Docker`](./docker.md#clean-up-docker).
