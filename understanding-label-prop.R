library(RSSL)

# Generate some data
data <- generateCrescentMoon(10, 2, 1)
data <- data[sample(1:nrow(data)), ]
rownames(data) <- NULL
data$Class[5:20] <- NA

colours <- as.integer(data$Class)
colours <- replace(colours, which(is.na(colours)), 3)
plot(data$X1, data$X2, col = colours + 5, asp = 1)



# Inputs
X   <- data[!is.na(data$Class), 2:3] # Feature matrix for labelled data
X_u <- data[is.na(data$Class), 2:3] # Feature matrix of unlabelled data
y   <- data$Class[!is.na(data$Class)] # Vector of labels

# Settings
k <- 5 # number of neighbours

# Source: https://github.com/jkrijthe/RSSL/blob/master/R/GRFClassifier.R

# Pre-processing
Xin <- rbind(X, X_u)

indicator <- as.integer(y) - 1
Y <- matrix(c(!indicator, indicator), ncol = 2)
colnames(Y) <- levels(y)


# Adjacency matrix
Ds <- as.matrix(dist(Xin))

# For every row of the distance matrix, sort the distances and get index of the
# k smallest distances(N.B. (2:k + 1) because the 1st position is the
# obsevrations distance to itself, i.e. zero). This produces a matrix of indices
# where each column is an observation and the k rows are the nearest neighbours.
# The matrix is then converted to an ordered vector of indices for k nearest
# neighbours of each observation.
neighbours <- apply(Ds, 1, function(x) sort(x, index.return = TRUE)$ix[2:(k + 1)])
neighbours <- as.integer(neighbours)

# Turn the neighbourhood vector into an n*n indicator matrix of the neighbours
# of each observation
neighbourhoods <- Matrix::sparseMatrix(i = rep(1:nrow(Xin), each = k), j = neighbours, x = 1, dims = c(nrow(Xin), nrow(Xin)))
neighbourhoods <- as.matrix(neighbourhoods)

# But we don't just want the direct neighbours of each observation, we want to
# know when an observation can be linked to other observations by traversing
# neighbourhoods. In essence, we want to know if an observation's neighbourhood
# is within the neighbourhood of another observation. So tranposing the
# neighbourhood matrix and doing an element-wise OR comparison get us the
# 'adjacency' (aka 'weight') graph in the form of a matrix.
# https://en.wikipedia.org/wiki/Adjacency_matrix
W <- (adj | t(adj)) * 1

# With the adjacency graph constructed we can now compute the solution to the
# 'harmonic function', which is a closed-form shortcut of the iterative label
# propagation algorithm.

l <- nrow(Y) # the number of labeled observations
n <- nrow(W) # total number of observations

# We need to construct the 'Laplacian matrix' of the graph: L = D - W
# D is the 'degree matrix'; the number of other observations which are linked to
# a given observation (in the form of a diagonal matrix).
# https://en.wikipedia.org/wiki/Degree_matrix
# https://en.wikipedia.org/wiki/Laplacian_matrix
D <- diag(colSums(W))
L <- D - W

# Then the solution to the harmonic funciton is computed (roughly speaking) as:
#  - (inverse of the portion of the Lapacian for only the unlabelled data)
#  - multiplied by (the portion of the Laplacian for the rows of the unlabelled
#       obsevrations by the columns of the labelled observations)
#  - multiplied by (the 2-column matrix of labels for the observations)
L_uu <- L[(l + 1):n, (l + 1):n]
L_ul <- L[(l + 1):n, 1:l]

solution <- -solve(L_uu, L_ul %*% Y)
# N.B. The solution can be interpreted as the probability of an unlabelled
# observation arriving at a labelled observation via a random walk on the graph.

# 'Class Mass Normalisation' heuristic which apparently gives better class
# predictions than just using the estimated probabilities alone.
q <- colSums(Y) + 1
cmn <- solution * matrix(rep(q / colSums(solution), n - l), nrow = n - l, byrow = TRUE)

probability <- solution
predictions <- matrixStats::rowRanks(cmn)
