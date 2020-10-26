import breeze.numerics._
import breeze.linalg._

/*
* Defines object that implements k-means algorithm for document clustering. Documents are assumed to be in the 
* columns of the input matrix
*/

object KMeans {

	def cluster(X: DenseMatrix[Double], k: Int) : DenseVector[Int] = {
		/*
		* Runs main clustering k-means algorithm
		*
		* Arguments:
		*			-X: Term-document matrix
		*			-k: Number of clusters
		*
		*  Returns: 
		*			-DenseVector[Int]: Document labels
		*/

		val shape = (X.rows, X.cols)
		var SSE_prev = 1000000000.0
		var SSE_curr = 999999999.0

		// Random initialization		
		var centroids = init_centroids(X, shape, k)
		var cluster_labels = assign_to_nearest_centroid(X, centroids)
		
		// Run KMeans
		while(SSE_prev > SSE_curr) {
			centroids = assign_centroids(X, k, cluster_labels)	
			cluster_labels = assign_to_nearest_centroid(X, centroids)
			SSE_prev = SSE_curr
			SSE_curr = calc_sse(X, centroids, cluster_labels, k)
		}		

		return cluster_labels

	} 	

	def init_centroids(X: DenseMatrix[Double], dim: (Int, Int), k: Int) : DenseMatrix[Double] = {
		/*
		* Returns vector of k random integers in the range [0, max_index)
		*
		* Arguments:
		*			- X: Term-by-document matrix
		*			- dim: Dimensions of X
		*			- k: Length of vector to return
		*
		* Returns:
		*			- DenseMatrix[Double]: Vector of random indeces in the range
		*/	

		var centroid_indices = DenseVector.rand(k)	

		do {	
			centroid_indices = DenseVector.rand(k)
			centroid_indices :*= dim._2.toDouble
			centroid_indices = floor(centroid_indices) 
		} while(!ensure_unique(centroid_indices))
	
		var centroids = DenseMatrix.zeros[Double](dim._1, k)
		for(i <- 0 until k)
			centroids(::, i) := X(::, centroid_indices(i).toInt)
		
		return centroids

	}

	def ensure_unique(vec: DenseVector[Double]) : Boolean = {
		/*
		* Ensures randomly generated indices are unique 
		*
		* Arguments: 
		*			-vec: Indices of initialized centroids
		*
		* Returns:
		*			-Boolean: True if unique
		*/

		var current_indices = Array.fill(vec.length){-1.0}

		for(i <- 0 until vec.length) {
			if(current_indices contains vec(i))
				return false
			else
				current_indices(i) = vec(i)		
		}		

		return true

	}

	def assign_to_nearest_centroid(X: DenseMatrix[Double], centroids: DenseMatrix[Double]) : DenseVector[Int] = {
		/*
		* Assigns documents to nearest centroid based on Euclidean Distance
		*
		* Arguments:
		*			- X: term-by-document matrix
		*			- centroids: Matrix where each row is a centroid
		*
		* Returns: 
		*			- DenseVector[Int]: Vector of document labels 
		*/		

		var cluster_labels = DenseVector.zeros[Int](X.cols)

		for(i <- 0 until X.cols) {
			var closest_index = 0
			var closest_dist = norm(X(::, i) - centroids(::, closest_index))

			for(j <- 1 until centroids.cols) {
				val new_dist = norm(X(::, i) - centroids(::, j))

				if(new_dist < closest_dist) {
					closest_dist = new_dist
					closest_index = j
				}	
			}
		
			cluster_labels(i) = closest_index
		}	

		return cluster_labels

	}

	def assign_centroids(X: DenseMatrix[Double], k: Int, cluster_labels: DenseVector[Int]) : DenseMatrix[Double] = {
		/*	
		* Calculates centroids (mean of document vectors in cluster)
		*
		* Arguments:
		*			- X: term-by-document matrix
		*			- k: Number of clusters
		*			- cluster_labels: labels of documents
		*
		* Returns:  
		*			- DenseMatrix[Double]: Matrix where each column is a centroid
		*/

		var cluster_sums = Array.fill(k){DenseVector.zeros[Double](X.rows)} //[DenseVector[Double]](k)
		for(i <- 0 until k)
			cluster_sums(i) = DenseVector.zeros[Double](X.rows)

		var cluster_counts = DenseVector.zeros[Double](k)
		
		for(i <- 0 until cluster_labels.length) {
			val cluster = cluster_labels(i)
			cluster_sums(cluster) :+= X(::, i) 
			cluster_counts(cluster) += 1
		}
		
		var centroids = DenseMatrix.zeros[Double](X.rows, k)
		for(i <- 0 until k)
			centroids(::, i) := cluster_sums(i) / cluster_counts(i)

		return centroids
	}

	def calc_sse(X: DenseMatrix[Double], centroids: DenseMatrix[Double], cluster_labels: DenseVector[Int], k: Int) : Double = {
		/*
		* Calculates sum of intra-cluster squared errors
		*
		* Arguments: 
		*			- X: term-by-document matrix
		*			- centroids: Matrix where each column is a centroid
		*			- cluster_labels: vector of document labels
		*
		* Returns: 		
		*			- Double: SSE
		*/

		var cluster_sse = DenseVector.zeros[Double](k)
		
		for(i <- 0 until cluster_labels.length) {
			val cluster = cluster_labels(i)
			cluster_sse(cluster) += norm(X(::, i) - centroids(::, cluster)) 
		}

		return sum(cluster_sse)

	}

}
