from Worker import Main


if __name__ == '__main__':
    main_class = Main(n_nodes=3)
    main_class.setup_atoms()

    main_class.run_PAC()
