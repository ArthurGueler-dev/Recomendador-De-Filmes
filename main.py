import pandas as pd
from sklearn.neighbors import NearestNeighbors
import tkinter as tk
from tkinter import ttk, messagebox

# Carregar dados e treinar modelo
movies_df = pd.read_csv('movies.csv', usecols=['movieId', 'title'], dtype={'movieId': 'int32', 'title': 'str'})
rating_df = pd.read_csv('ratings_with_names.csv', usecols=['userId', 'movieId', 'rating'],
                        dtype={'userId': 'object', 'movieId': 'int32', 'rating': 'float32'})

df = pd.merge(rating_df, movies_df, on='movieId')
movie_features_df = df.pivot_table(index='userId', columns='title', values='rating').fillna(0)

# Treinar modelo Nearest Neighbors
model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
model_knn.fit(movie_features_df)

# Função para recomendar filmes com base no nome do usuário
def recommend_movies_for_user(username, num_recommendations=5):
    try:
        # Verificar se o usuário está presente no índice
        if username not in movie_features_df.index:
            raise KeyError(f"Usuário '{username}' não encontrado na base de dados.")

        # Consultar usando o índice numérico do usuário
        query_index = movie_features_df.index.get_loc(username)
        distances, indices = model_knn.kneighbors(movie_features_df.iloc[query_index, :].values.reshape(1, -1),
                                                  n_neighbors=num_recommendations + 1)

        # Obter os índices dos filmes recomendados (excluindo o próprio usuário)
        recommended_indices = indices.flatten()[1:num_recommendations + 1]

        # Recuperar os títulos dos filmes recomendados
        recommended_movies = movie_features_df.columns[recommended_indices].tolist()

        # Recuperar os filmes assistidos e avaliados pelo usuário
        filmes_avaliados = movie_features_df.columns[movie_features_df.loc[username] > 0].tolist()

        # Mostrar histórico de filmes avaliados pelo usuário
        ds_usuario = movie_features_df.loc[username][movie_features_df.loc[username] > 0].sort_values(ascending=False)

        return recommended_movies, filmes_avaliados, ds_usuario

    except KeyError as e:
        messagebox.showerror("Erro", str(e))
        return [], [], pd.Series()

# Função para atualizar a interface com as recomendações e o histórico do usuário
def update_user_data(event):
    selected_user = user_combobox.get()
    if selected_user:
        recommended_movies, filmes_avaliados, historico_usuario = recommend_movies_for_user(selected_user)

        recommendations_text.set('\n'.join(recommended_movies) if recommended_movies else 'Nenhuma recomendação disponível')
        
        if not historico_usuario.empty:
            history_lines = [f"{title}: {rating}" for title, rating in historico_usuario.items()]
            history_text.set('\n'.join(history_lines))
        else:
            history_text.set('Nenhum histórico disponível')

# Criar a interface gráfica
root = tk.Tk()
root.title("Sistema de Recomendação de Filmes")

frame = ttk.Frame(root, padding="10")
frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

# Combobox para seleção do usuário
user_label = ttk.Label(frame, text="Selecione o Usuário:")
user_label.grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)

user_combobox = ttk.Combobox(frame, values=movie_features_df.index.tolist(), state='readonly')
user_combobox.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
user_combobox.bind('<<ComboboxSelected>>', update_user_data)

# Text widget para exibir recomendações
recommendations_label = ttk.Label(frame, text="Recomendações:")
recommendations_label.grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)

recommendations_text = tk.StringVar()
recommendations_display = ttk.Label(frame, textvariable=recommendations_text, wraplength=400)
recommendations_display.grid(row=2, column=0, columnspan=2, padx=5, pady=5, sticky=tk.W)

# Frame para exibir histórico com barra de rolagem
history_label = ttk.Label(frame, text="Histórico de Avaliações:")
history_label.grid(row=3, column=0, padx=5, pady=5, sticky=tk.W)

history_frame = ttk.Frame(frame)
history_frame.grid(row=4, column=0, columnspan=2, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))

history_canvas = tk.Canvas(history_frame)
history_scrollbar = ttk.Scrollbar(history_frame, orient="vertical", command=history_canvas.yview)
history_content = ttk.Frame(history_canvas)

history_content.bind(
    "<Configure>",
    lambda e: history_canvas.configure(
        scrollregion=history_canvas.bbox("all")
    )
)

history_canvas.create_window((0, 0), window=history_content, anchor="nw")
history_canvas.configure(yscrollcommand=history_scrollbar.set)

history_canvas.pack(side="left", fill="both", expand=True)
history_scrollbar.pack(side="right", fill="y")

history_text = tk.StringVar()
history_display = ttk.Label(history_content, textvariable=history_text, wraplength=400)
history_display.pack(side="top", fill="both", expand=True)

root.mainloop()