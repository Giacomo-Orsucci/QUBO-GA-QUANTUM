from qat.qlmaas.connection import QLMaaSConnection

print("Connessione in corso...")
connection = QLMaaSConnection()

print("\nConnessione stabilita! Interrogo il server...")
available_qpus = connection.get_qpus()

print("\n=== LISTA DELLE MACCHINE/EMULATORI DISPONIBILI ===")
for qpu in available_qpus:
    print(f"- {qpu}")
print("==================================================")