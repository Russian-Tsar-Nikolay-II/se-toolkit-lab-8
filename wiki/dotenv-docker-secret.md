# `.env.docker.secret`

<h2>Table of contents</h2>

- [What is `.env.docker.secret`](#what-is-envdockersecret)
- [`REGISTRY_PREFIX`](#registry_prefix)
- [`backend`](#backend)
  - [`BACKEND_NAME`](#backend_name)
  - [`BACKEND_DEBUG`](#backend_debug)
  - [`BACKEND_RELOAD`](#backend_reload)
  - [`BACKEND_CONTAINER_ADDRESS`](#backend_container_address)
  - [`BACKEND_CONTAINER_PORT`](#backend_container_port)
  - [`BACKEND_HOST_ADDRESS`](#backend_host_address)
  - [`BACKEND_HOST_PORT`](#backend_host_port)
  - [`BACKEND_ENABLE_INTERACTIONS`](#backend_enable_interactions)
  - [`BACKEND_ENABLE_LEARNERS`](#backend_enable_learners)
- [`postgres`](#postgres)
  - [`POSTGRES_DB`](#postgres_db)
  - [`POSTGRES_USER`](#postgres_user)
  - [`POSTGRES_PASSWORD`](#postgres_password)
  - [`POSTGRES_HOST_ADDRESS`](#postgres_host_address)
  - [`POSTGRES_HOST_PORT`](#postgres_host_port)
- [`pgadmin`](#pgadmin)
  - [`PGADMIN_EMAIL`](#pgadmin_email)
  - [`PGADMIN_PASSWORD`](#pgadmin_password)
  - [`PGADMIN_HOST_ADDRESS`](#pgadmin_host_address)
  - [`PGADMIN_HOST_PORT`](#pgadmin_host_port)
- [`caddy`](#caddy)
  - [`CADDY_CONTAINER_PORT`](#caddy_container_port)
- [`Qwen Code` API](#qwen-code-api)
  - [`QWEN_CODE_API_URL`](#qwen_code_api_url)
- [LMS](#lms)
  - [`LMS_API_HOST_ADDRESS`](#lms_api_host_address)
  - [`LMS_API_HOST_PORT`](#lms_api_host_port)
  - [`LMS_API_KEY`](#lms_api_key)
- [`Autochecker` API](#autochecker-api)
  - [`AUTOCHECKER_API_URL`](#autochecker_api_url)
  - [`AUTOCHECKER_API_LOGIN`](#autochecker_api_login)
  - [`AUTOCHECKER_API_PASSWORD`](#autochecker_api_password)
- [Constants](#constants)
  - [`CONST_POSTGRESQL_SERVICE_NAME`](#const_postgresql_service_name)
  - [`CONST_POSTGRESQL_SERVER_NAME`](#const_postgresql_server_name)
  - [`CONST_POSTGRESQL_DEFAULT_PORT`](#const_postgresql_default_port)
  - [`CONST_PGADMIN_CONTAINER_PORT`](#const_pgadmin_container_port)

## What is `.env.docker.secret`

`.env.docker.secret` is a [`.env` file](./environments.md#env-file) that stores [environment variables](./environments.md#environment-variable) for [`Docker Compose`](./docker-compose.md#what-is-docker-compose).

The values are substituted into [`docker-compose.yml`](../docker-compose.yml) when running commands with the `--env-file` flag (e.g., `docker compose --env-file .env.docker.secret up --build`).

Therefore, use the values from the `.env.docker.secret` file stored on the [machine](./computer-networks.md#machine) where you deployed via `Docker Compose`.

The default values are in [`.env.docker.example`](../.env.docker.example).

> [!NOTE]
> `.env.docker.secret` was added to [`.gitignore`](./git.md#gitignore) because you may specify there
> [secrets](./environments.md#secrets) such as the [LMS API key](./lms-api.md#lms-api-key) or the [address of your VM](./vm.md#your-vm-ip-address).

## `REGISTRY_PREFIX`

The prefix prepended to [Docker image](./docker.md#image) names when pulling base images.
By default, it points to a [`Harbor`](https://goharbor.io/) cache proxy on the university network, which avoids [`DockerHub`](./docker.md#dockerhub) rate limits.
Outside the university network, set it to an empty string to pull directly from `DockerHub`.

This value is used as a build argument in the [`backend`](#backend) and [`caddy`](#caddy) service [`Dockerfile`](./docker.md#what-is-docker) builds, and as an image prefix for the [`postgres`](#postgres) and [`pgadmin`](#pgadmin) services in [`docker-compose.yml`](../docker-compose.yml).

Default: `harbor.pg.innopolis.university/docker-hub-cache/`

## `backend`

Variables for the [`backend` service](./docker-compose-yml.md#backend-service).

### `BACKEND_NAME`

The display name of the application.

Default: `"Learning Management Service"`

### `BACKEND_DEBUG`

Enables debug mode in the [web server](./web-infrastructure.md#web-server). When `true`, the server returns detailed error messages.

Default: `false`

### `BACKEND_RELOAD`

Enables auto-reload. When `true`, the [web server](./web-infrastructure.md#web-server) restarts automatically when source files change.

Default: `false`

### `BACKEND_CONTAINER_ADDRESS`

The [IP address](./computer-networks.md#ip-address) the backend [listens on](./computer-networks.md#listen-on-a-port) inside the [container](./docker.md#container). [`0.0.0.0`](./computer-networks.md#0000) means all network interfaces.

Default: `0.0.0.0`

### `BACKEND_CONTAINER_PORT`

The [port number](./computer-networks.md#port-number) the backend [listens on](./computer-networks.md#listen-on-a-port) inside the [container](./docker.md#container).

Default: `8000`

### `BACKEND_HOST_ADDRESS`

The [IP address](./computer-networks.md#ip-address) exposed on the [host](./computer-networks.md#host). [`127.0.0.1`](./computer-networks.md#127001) restricts access to the local machine only.

Default: `127.0.0.1`

### `BACKEND_HOST_PORT`

The [port number](./computer-networks.md#port-number) exposed on the [host](./computer-networks.md#host) for accessing the backend.

Default: `42001`

### `BACKEND_ENABLE_INTERACTIONS`

A [feature flag](./environments.md#feature-flag) for enabling the `/interactions` endpoint.

Default: `true`

### `BACKEND_ENABLE_LEARNERS`

A [feature flag](./environments.md#feature-flag) for enabling the `/learners` endpoint.

Default: `true`

## `postgres`

Variables for the [`postgres` service](./docker-compose-yml.md#postgres-service).

### `POSTGRES_DB`

The name of the [database](./database.md#what-is-a-database) created on the first startup.

Default: `db-lab-7`

### `POSTGRES_USER`

The username for the [`PostgreSQL`](./database.md#postgresql) database.

Default: `postgres`

### `POSTGRES_PASSWORD`

The password for the [`PostgreSQL`](./database.md#postgresql) database.

Default: `postgres`

### `POSTGRES_HOST_ADDRESS`

The [IP address](./computer-networks.md#ip-address) exposed on the [host](./computer-networks.md#host). [`127.0.0.1`](./computer-networks.md#127001) restricts access to the local machine only.

Default: `127.0.0.1`

### `POSTGRES_HOST_PORT`

The [port number](./computer-networks.md#port-number) exposed on the [host](./computer-networks.md#host) for accessing [`PostgreSQL`](./database.md#postgresql).

Default: `42004`

## `pgadmin`

Variables for the [`pgadmin` service](./docker-compose-yml.md#pgadmin-service).

### `PGADMIN_EMAIL`

The email used to log in to [`pgAdmin`](./pgadmin.md#what-is-pgadmin).

Default: `admin@example.com`

### `PGADMIN_PASSWORD`

The password used to log in to [`pgAdmin`](./pgadmin.md#what-is-pgadmin).

Default: `admin`

### `PGADMIN_HOST_ADDRESS`

The [IP address](./computer-networks.md#ip-address) exposed on the [host](./computer-networks.md#host).

[`127.0.0.1`](./computer-networks.md#127001) restricts access to the local machine only.

Default: `127.0.0.1`

### `PGADMIN_HOST_PORT`

The [port number](./computer-networks.md#port-number) exposed on the [host](./computer-networks.md#host) for accessing [`pgAdmin`](./pgadmin.md#what-is-pgadmin).

Default: `42003`

## `caddy`

Variables for the [`caddy` service](./docker-compose-yml.md#caddy-service).

### `CADDY_CONTAINER_PORT`

The [port number](./computer-networks.md#port-number) that [`Caddy`](./caddy.md#what-is-caddy) [listens on](./computer-networks.md#listen-on-a-port) inside the [container](./docker.md#container).

Default: `80`

## `Qwen Code` API

Variable for the [`Qwen Code` API](./qwen-code-api.md#what-is-qwen-code-api) used by the [`caddy` service](./docker-compose-yml.md#caddy-service).

### `QWEN_CODE_API_URL`

The URL of the [`Qwen Code` API](./qwen-code-api.md#what-is-qwen-code-api) that [`Caddy`](./caddy.md#what-is-caddy) reverse-proxies requests to.

Default: `http://qwen-code-api:8080`

## LMS

Variables for the [LMS API](./lms-api.md#about-the-lms-api).

### `LMS_API_HOST_ADDRESS`

The [IP address](./computer-networks.md#ip-address) of the [LMS API](./lms-api.md#about-the-lms-api) exposed on the [host](./computer-networks.md#host).

[`0.0.0.0`](./computer-networks.md#0000) accepts connections from any network interface.

Default: `0.0.0.0`

### `LMS_API_HOST_PORT`

The [LMS API host port](./lms-api.md#lms-api-host-port).

Default: `42002`

### `LMS_API_KEY`

The [LMS API key](./lms-api.md#lms-api-key).

Default: `my-secret-api-key`

## `Autochecker` API

Variables for the [autochecker](./autochecker.md#what-is-the-autochecker) ETL pipeline.

### `AUTOCHECKER_API_URL`

The base URL of the autochecker API.

Default: `https://auche.namaz.live`

### `AUTOCHECKER_API_LOGIN`

The login used to authenticate with the autochecker API.

Default: [`<autochecker-api-login>`](./autochecker-api.md#autochecker-api-login-placeholder)

### `AUTOCHECKER_API_PASSWORD`

The password used to authenticate with the autochecker API.

Default: [`<autochecker-api-password>`](./autochecker-api.md#autochecker-api-password-placeholder)

## Constants

Values that should not be changed.
They are defined here for convenient referencing in [`docker-compose.yml`](../docker-compose.yml).

### `CONST_POSTGRESQL_SERVICE_NAME`

The [`Docker Compose` service name](./docker-compose.md#service-name) for [`PostgreSQL`](./database.md#postgresql).
Other [services](./docker-compose.md#service) use this name to connect to the database via [`Docker Compose` networking](./docker-compose.md#docker-compose-networking).

Default: `postgres`

### `CONST_POSTGRESQL_SERVER_NAME`

The display name for the [`PostgreSQL`](./database.md#postgresql) server in [`pgAdmin`](./pgadmin.md#what-is-pgadmin).

Default: `postgres-lab-7`

### `CONST_POSTGRESQL_DEFAULT_PORT`

The default [port number](./computer-networks.md#port-number) [`PostgreSQL`](./database.md#postgresql) [listens on](./computer-networks.md#listen-on-a-port) inside the [container](./docker.md#container).

Default: `5432`

### `CONST_PGADMIN_CONTAINER_PORT`

The [port number](./computer-networks.md#port-number) that [`pgAdmin`](./pgadmin.md#what-is-pgadmin) is [listening on](./computer-networks.md#listen-on-a-port) inside the [container](./docker.md#container).

Default: `80`
