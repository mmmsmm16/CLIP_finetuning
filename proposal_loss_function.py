import torch

def proposed_clip_loss(T, I):
    """
    T: テキストの埋め込みを含むテンソル [batch_size, embedding_size]
    I: 画像の埋め込みを含むテンソル [batch_size, embedding_size]
    """
    
    # ノルムで正規化（コサイン類似度の計算のため）
    T_normalized = T / T.norm(dim=1, keepdim=True)
    I_normalized = I / I.norm(dim=1, keepdim=True)
    
    # コサイン類似度の計算
    T_dot_T = torch.mm(T_normalized, T_normalized.t())
    I_dot_T = torch.mm(I_normalized, T_normalized.t())
    T_dot_I = torch.mm(T_normalized, I_normalized.t())
    
    # L_{I,T} と L_{T,I} の計算
    L_I_T = torch.abs(T_dot_T - I_dot_T)
    L_T_I = torch.abs(T_dot_T - T_dot_I)
    
    # 各要素の誤差の合計を計算
    loss = (L_I_T.sum() + L_T_I.sum()) / 2
    
    return loss
 
