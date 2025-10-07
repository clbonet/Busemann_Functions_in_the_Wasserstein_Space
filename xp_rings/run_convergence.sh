wgf() {
       	method=$1

       	python convergence_rings.py \
        --distance $method \
        --n_try 100
}

for method in "sotdd" "swb1dg" "swbg"
do
  	wgf $method
done
