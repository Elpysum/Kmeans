###################################### Code Kmeans - Anas et Luana


###################################### #La fonction Kmeans

mykmeans<-function(nbr.classe,data){
  
  #Initialisation des centroides(1)
  centroides <- matrix(NA,nbr.classe,ncol(data))
  rownames(centroides) <- paste("centre",1:nbr.classe,sep = "_")
  colnames(centroides) <- colnames(data)
  
  centroides <- as.matrix(data[sample(nrow(data),nbr.classe),])  
  
  #Calcul des distances euclidiennes(2) 
  dist.eucl <- function(centroides){
    Dist <- matrix( NA , nrow = nrow(data) , ncol = nbr.classe)
    
    for (i in 1:nrow(data))
      for (j in 1:nbr.classe)
        Dist[i,j] <- sqrt(sum((data[i,] - centroides[j,])^2))
      
      return(Dist)
  }
  
  
  while (TRUE){
    #Classiffication(3)
    clusters <- apply(dist.eucl(centroides), 1, which.min)
    
    test.centroides <- centroides
    
    #Mise a jour centroides(4)
    for (i in 1:nrow(centroides))
      centroides[i,] <- colMeans(data[clusters==i,])
    
    if (sum((test.centroides - centroides)^2) < 10^-6)
      break
  }
  
  #Comptage des chiffres
  Partition <- table(clusters) 
  rownames(Partition) <- c(paste(0:9))
  
  return(list(clusters = clusters, partition = Partition)) 
}

########################################## #Application sur MNIST

load("C:/Users/Luana Paully/Downloads/MNIST.RData")
ls()

#Dimensions
dim(x_test) 
dim(x_train) 

#Petit exemple
image(matrix(x_train[25,],28,28)[,28:1], col=rev(grey(0:255/255)))

#Classification de Kmeans(1) - TRAIN
toto <- mykmeans(10,x_train)
toto$partition
#Matrice de confusion de nos classes de 'train'(2)
MC_train <- table(toto$clusters, as.factor(y_train))
rownames(MC_train) <- c(paste(0:9))
#Pourcentage d'efficacité de mykmeans pour 'train'
perc <- 100*(sum(apply(MC_train,2,max) / colSums(MC_train)) / 10 )
print(MC_train)
print(perc)

#Classification de Kmeans(1) - TEST
toto <- mykmeans(10,x_test)
toto$partition
#Matrice de confustion sur le jeu de données 'test'(2)
MC_test <- table(toto$clusters, as.factor(y_test) ) 
rownames(MC_test) <- c(paste(0:9))
#Pourcentage d'efficacité de mykmeans pour 'test'
perc <- 100*(sum(apply(MC_test,2,max) / colSums(MC_test)) / 10 )
print(MC_test)
print(perc)


############################################ # Comparaison avec Kmeans "normal"

real_kmeans=kmeans(x_test,10)
inertie_interne_kmeans <- 100 * (real_kmeans$betweenss / real_kmeans$totss )
inertie_interne_kmeans

#Matrice de confusion du vrai kmeans
MC_kmeans=table(real_kmeans$cluster, y_test)
rownames(MC_kmeans)=c(paste(0:9))
MC_kmeans
accuracy_perc <- 100*(sum(apply(MC_kmeans,2,max) / colSums(MC_kmeans)) / 10 )
accuracy_perc


########################################## # Reseau de neurones 

# A installer si pas fait avant

# install.packages("keras")
# install.packages("caret")
# install.packages("nnet")
# install_keras()
# install_keras(tensorflow = "gpu")

library(keras)
library(caret)
library(nnet)
mnist <- dataset_mnist()

#Creation des variables pour les donnees train et test:
train_images <- mnist$train$x
train_labels <- mnist$train$y
test_images <- mnist$test$x
test_labels <- mnist$test$y

#on convertit les tableaux 3D ( images , largeur , longueur) en matrice 2D en restructurant la longueur 
# et largeur en une seule dimension
# c-a-d les images 28*28 en vecteurs 784. Ensuite, on convertit les niveaux de gris en réels entre 0 et 1.
train_images <- array_reshape(train_images, c(nrow(train_images), 28*28) ) / 255
test_images <- array_reshape(test_images, c(nrow(x_test), 28*28) ) / 255

#On utilise la fonction to_categorical de Keras 
#pour convertir les vecteurs d'entiers entre 0 et 9 en matrices de classes binaires.
train_labels <- to_categorical(train_labels, 10)
test_labels <- to_categorical(test_labels, 10)


network <- keras_model_sequential() %>%
  layer_dense(units = 256, activation = "relu", input_shape = c(784)) %>% 
  layer_dropout(rate = 0.4 ) %>%
  layer_dense(units = 128, activation = "relu") %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 10, activation = "softmax")
summary(network)



network %>% compile(
  optimizer = "rmsprop",
  loss = "categorical_crossentropy",
  metrics = c("accuracy")
)


#on utilise la fonction fit() pour entrainer le modele en une periode de longueur 30 avec 128 images
network_training <- network %>% fit(train_images, train_labels, epochs = 30, 
                                    batch_size = 128, validation_split = 0.2)


#Metriques de la perte et la precision
plot(network_training) 
metrics <- network %>% evaluate(test_images, test_labels)
metrics


#Genere les predictions sur 'test' (3)
prediction <- network %>% predict_classes(test_images)
prediction[1:100]


affiche_chiffre <- function(X) {
  m <- matrix(unlist(X),nrow = 28,byrow = T)  
  m <- t(apply(m, 2, rev))
  image(m,col=grey.colors(255))
} 

#Ligne 1 - colonne 1
affiche_chiffre(test_images[1, ]) 


#100 images pour une meilleure visualisation(3)
par(mfcol = c(10, 10), mar = rep(0, 4))
for (label in 0:9){
  class.idx <- which(prediction == label)
  
  for (i in sample(class.idx, 10))
    image(matrix(test_images[i,], 28)[,28:1], col = rev(grey(0:255/255)),
          axes = FALSE, bty = "n")
}


###################################### # CAH - laisse a la fin - cela prend beaucoup de temps a compiler 

library(NbClust)

kmeans_et_CAH <- function( data_images , min_nc=6,max_nc=15, acp =FALSE) {
  
  res = list()
  indices_vector = c("silhouette")
  
  if ( acp == TRUE) {
    pca = PCA( data_images , ncp = 2 )
    data_images = pca$ind$coord
  }
  
  res$mykmeans = NbClust( data=data_images , min.nc = min_nc , max.nc = max_nc ,
                          method = "kmeans" , index = indices_vector )
  
  res$kmeans = NbClust( data=data_images , min.nc = min_nc , max.nc = max_nc , 
                        method = "kmeans" , index = indices_vector )
  
  res$cah_complete = NbClust( data=data_images , min.nc = min_nc ,max.nc = max_nc , 
                              method = "complete" , index = indices_vector )
  
  res$cah_ward = NbClust( data=data_images , min.nc = min_nc , max.nc = max_nc , 
                          method = "ward.D2" , index = indices_vector )
  
  return(res)
}