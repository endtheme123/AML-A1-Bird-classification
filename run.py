from model.baseline import BirdClassifier
from trainer import Trainer

def main(args):
    
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu"
    )
    print("Cuda available ?", torch.cuda.is_available())
    print("Pytorch device:", device)
    seed = 11
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    
    train_dataloader, train_dataset = get_dataloader(args)
    


    model = BirdClassifier()
    model.to(device)

    trainer = Trainer(model, )

    

if __name__ == "__main__":
    args = parse_args()
    main(args)