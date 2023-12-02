az acr build --file docker/DependencyDockerfile --registry sagarmlops23 --image base .
az acr build --file docker/FinalDockerfile --registry sagarmlops23 --image digits .