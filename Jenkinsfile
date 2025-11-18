pipeline {
    agent any

    // Define environment variables (available in all stages)
    environment {
        APP_VERSION = "1.0.5"
        DEPLOY_ENV = "production"
    }

    // Define build tools (Maven in this case)
    tools {
        maven "Maven_3.9"   // this name must match the Maven installation name in Jenkins
    }

    stages {

        stage('Build') {
            steps {
                echo "Building application version: ${APP_VERSION}"
                echo "Environment: ${DEPLOY_ENV}"

                // Using Maven installed via tools{}
                sh "mvn --version"
                sh "mvn clean install"
            }
        }

        stage('Test') {
            steps {
                echo 'Testing..'
                // test commands here...
            }
        }

        stage('Deploy') {
            steps {
                echo "Deploying version ${APP_VERSION} to ${DEPLOY_ENV}"
                // deployment commands...
            }
        }
    }

    post {
        always {
            echo 'Post build condition running'
        }
        success {
            echo 'This runs only if the build succeeded'
        }
        failure {
            echo 'Post action if build failed'
        }
        unstable {
            echo 'This runs if the build is unstable'
        }
    }
}
