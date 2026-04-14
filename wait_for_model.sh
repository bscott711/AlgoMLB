while ! grep -q "Production model saved" .training_pa_v1.1.log; do
    sleep 30
    echo "Waiting for model to complete..."
done
echo "Model completed!"
