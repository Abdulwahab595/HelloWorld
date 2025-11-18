pipeline {
    agent any

    parameters {
        string(name: 'APP_ENV', defaultValue: 'dev', description: 'Application environment')
        choice(name: 'BUILD_TYPE', choices: ['Debug', 'Release'], description: 'Build Type')
        booleanParam(name: 'EXECUTE_TESTS', defaultValue: true, description: 'Run Test Stage?')
    }

    stages {
        stage('Build') {
            steps {
                echo "Building the project..."
                echo "Environment: ${params.APP_ENV}"
                echo "Build Type: ${params.BUILD_TYPE}"
            }
        }

        stage('Test') {
            when {
                expression { return params.EXECUTE_TESTS == true }
            }
            steps {
                echo 'Running tests...'
                // Add test commands here
            }
        }

        stage('Deploy') {
            steps {
                echo "Deploying application..."
                // Add deployment commands here
            }
        }
    }
}
